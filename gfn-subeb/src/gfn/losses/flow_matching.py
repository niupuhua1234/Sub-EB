

import torch
from torchtyping import TensorType

from src.gfn.containers.trajectories import Trajectories
from src.gfn.containers.states import States
from src.gfn.losses.base import  StateDecomposableLoss,FParametrization
from src.gfn.estimators import LogitPBEstimator

# Typing
ScoresTensor = TensorType["n_states", float]
LossTensor = TensorType[0, float]

class FMParametrization(FParametrization):
    r"""
    :math:`\mathcal{O}_{edge}` is the set of functions from the non-terminating edges
    to :math:`\mathbb{R}^+`. Which is equivalent to the set of functions from the internal nodes
    (i.e. without :math:`s_f`) to :math:`(\mathbb{R})^{n_actions}`, without the exit action (No need for
    positivity if we parametrize log-flows).
    """
    def __init__(self, logF,logit_PB:LogitPBEstimator):
        self.logit_PB=logit_PB
        super().__init__(logF)
class FlowMatching(StateDecomposableLoss):
    def __init__(self, parametrization: FMParametrization,optimizer: torch.optim.Optimizer,) -> None:
        "alpha is the weight of the reward matching loss"
        self.env = parametrization.logF.env
        self.epsilon=1e-6
        super().__init__(parametrization,)#fill_value=-float('inf'))
        self.optimizer=optimizer

    def flow_matching_loss(self, states: States) -> ScoresTensor:
        """
        Compute the FM for the given states, defined as the log-sum incoming flows minus log-sum outgoing flows.
        The states should not include s0. The batch shape should be (n_states,).

        As of now, only discrete environments are handled.
        """

        assert len(states.batch_shape) == 1
        assert not torch.any(states.is_initial_state)

        inter_maskings,termi_maskings=states.forward_masks[...,:-1],states.forward_masks[...,-1]
        outgoing_log_flows = torch.full_like(states.forward_masks, self.fill_value, dtype=torch.float)
        outgoing_log_flows[...,:-1][inter_maskings] = self.parametrization.logF(states)[...,:-1][inter_maskings]
        outgoing_log_flows[..., -1][termi_maskings] = self.env.log_reward(states[termi_maskings])
        ###########################
        incoming_log_flows = torch.full_like(states.backward_masks, self.fill_value, dtype=torch.float)
        all_parent_states = self.env.all_step(states,Backward=True)
        all_actions       = torch.arange(self.env.bction_space.n).repeat(*states.batch_shape,1)
        repeat_states     = self.env.States(states.states_tensor.unsqueeze(-2).expand_as(all_parent_states.states_tensor))
        valid_actions  = self.env.bction2action(repeat_states[states.backward_masks],all_actions[states.backward_masks])
        valid_flow_all = self.parametrization.logF(all_parent_states[states.backward_masks])

        incoming_log_flows[states.backward_masks] =self.action_flow_gather(valid_flow_all,valid_actions)
        log_incoming_flows = torch.log(incoming_log_flows.exp().sum(dim=-1)+self.epsilon)
        log_outgoing_flows =  torch.log(outgoing_log_flows.exp().sum(dim=-1)+self.epsilon)
        return (log_incoming_flows - log_outgoing_flows).pow(2).mean()

    def reward_matching_loss(self, terminating_states: States) -> LossTensor:
        log_edge_flows = self.parametrization.logF(terminating_states)
        terminating_log_edge_flows = log_edge_flows[:, -1]
        log_rewards = self.env.log_reward(terminating_states)
        return (terminating_log_edge_flows - log_rewards).pow(2).mean()

    def update_model(self,trajectories: Trajectories,**kwargs):
        loss=self.__call__(trajectories)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def __call__(self, trajectories: Trajectories) -> LossTensor:
        states,last_states=trajectories.intermediary_states, trajectories.last_states
        fm_loss = self.flow_matching_loss(states)
        tm_loss=self.reward_matching_loss(last_states)
        return fm_loss+tm_loss
