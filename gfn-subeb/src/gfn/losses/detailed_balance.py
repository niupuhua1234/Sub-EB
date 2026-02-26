from dataclasses import dataclass

import torch
from torchtyping import TensorType

from src.gfn.containers.trajectories import Trajectories
from src.gfn.containers import Transitions
from src.gfn.estimators import LogStateFlowEstimator
from src.gfn.losses.base import EdgeDecomposableLoss, PFBasedParametrization

# Typing
ScoresTensor = TensorType["n_transitions", float]
LossTensor = TensorType[0, float]


class DBParametrization(PFBasedParametrization):
    r"""
    Corresponds to  :math:`\mathcal{O}_{PF} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3`, where
    :math:`\mathcal{O}_1` is the set of functions from the internal states (no :math:`s_f`)
    to :math:`\mathbb{R}^+` (which we parametrize with logs, to avoid the non-negativity constraint),
    and :math:`\mathcal{O}_2` is the set of forward probability functions consistent with the DAG.
    :math:`\mathcal{O}_3` is the set of backward probability functions consistent with the DAG, or a singleton
    thereof, if ``self.logit_PB`` is a fixed LogitPBEstimator.
    Useful for the Detailed Balance Loss.

    """
    def __init__(self, logit_PF,logit_PB, logF: LogStateFlowEstimator):
        self.logF=logF
        super().__init__(logit_PF,logit_PB)

class DetailedBalance(EdgeDecomposableLoss):
    """
        Forward     F(s0)P(·|s0)  ...  F(sn-1)P(sT|sn-1) → F(sT)P(sf|sT)
        ----------------------------------------------------------------
        Backward    F(s1)P(s0|s1)  ...  F(sT)P(sn-1|sT)  ← F(sf)P(sT|sf)
                      =1                                       R(sT)
        When not all states can be terminating states
        Forward     F(s0)P(·|s0)  ...  F(sn-1)P(sT|sn-1) → F(sT)P(sf|sT)
                                                                   =1
        ----------------------------------------------------------------
        Backward    F(s1)P(s0|s1)  ...  F(sT)P(sn-1|sT)  ← F(sf)P(sT|sf)
                      =1                                       R(sT)
    """

    def __init__(self, parametrization: DBParametrization,optimizer: torch.optim.Optimizer,all_sf=False):
        "If on_policy is True, the log probs stored in the transitions are used."
        super().__init__(parametrization)#,fill_value=-float('inf'))
        self.optimizer=optimizer
        self.all_sf=all_sf

    def get_scores(self, transitions: Transitions):
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")
        #####################
        # Forward passing   #
        #####################
        # automatically removes invalid transitions (i.e.  s_f->s_f)
        valid_index=~transitions.is_sink_action
        states = transitions.states[valid_index]
        actions = transitions.actions[valid_index]
        # uncomment next line for debugging
        # assert transitions.states.is_sink_state.equal(transitions.actions == -1)
        if states.batch_shape != tuple(actions.shape):
            raise ValueError("Something wrong happening with log_pf evaluations")
        valid_log_pf_all=self.forward_log_prob(states)
        valid_log_pf_actions= self.action_prob_gather(valid_log_pf_all,actions)  # pF(s->s')=p(·|s,a), T(s,a)=s'
        valid_log_F_s = self.parametrization.logF(states).squeeze(-1) # F is a flow functon, need not to be prob
        preds = valid_log_pf_actions + valid_log_F_s
        ########################
        #   Back_ward passinxg  #
        ########################
        # automatically removes the last&invalid transitions (i.e. s_T->s_f and s_f->s_f)
        internal_index= valid_index &~transitions.is_terminating_action
        terminal_index= valid_index &transitions.is_terminating_action
        valid_next_states = transitions.next_states[internal_index]
        non_exit_bctions  = transitions.env.action2bction(transitions.states[internal_index],actions[internal_index])
        # uncomment next line for debugging
        # assert transitions.next_states.is_sink_state.equal(transitions.is_done)
        # assert torch.all(transitions.is_terminating_action,transitions.is_done)
        valid_log_pb_all= self.backward_log_prob(valid_next_states)     #   pB(s->s')=p(s|s',a), T(s,a)=s'
        valid_log_pb_actions=self.action_prob_gather(valid_log_pb_all,non_exit_bctions)
        valid_log_F_s_next = self.parametrization.logF(valid_next_states).squeeze(-1)
        targets = torch.zeros_like(preds)
        targets[internal_index] = valid_log_pb_actions + valid_log_F_s_next
        targets[terminal_index]  = transitions.log_rewards[transitions.is_terminating_action]  #  F(s_f)pB(s_f->s_T)= F(s_T,s_f)=R(s_T)
        scores = preds - targets
        return scores

    def update_model(self,trajectories: Trajectories,**kwargs):
        transitions= trajectories.to_transitions()
        loss=self.__call__(trajectories)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def __call__(self,trajectories: Trajectories) -> LossTensor:
        transitions= trajectories.to_transitions()
        if not self.all_sf:
            scores = self.get_scores(transitions)
        else:
            scores =  self.get_modified_scores(transitions)
        loss = torch.sum(scores**2)/transitions.n_trajectories
        #loss=(scores**2).mean()
        if torch.isnan(loss):raise ValueError("loss is nan")
        return loss

    def get_modified_scores(self, transitions: Transitions) -> ScoresTensor:
        "DAG-GFN-style detailed balance, for when all states are connected to the sink"
        if transitions.is_backward:
            raise ValueError("Backward transitions are not supported")
        # automatically removes invalid transitions (i.e. s_T->s_f s_f->s_f)
        valid_index       =~transitions.is_sink_action &~transitions.is_terminating_action
        states            = transitions.states[valid_index]
        valid_next_states = transitions.next_states[valid_index]
        actions           = transitions.actions[valid_index]
        all_log_rewards   = transitions.all_log_rewards[valid_index]
        ########################
        #   for_ward passing  #
        ########################
        valid_log_pf_all=self.forward_log_prob(states)
        valid_log_pf_actions=self.action_prob_gather( valid_log_pf_all,actions)
        valid_log_pf_s_exit=valid_log_pf_all[...,-1]
        # The following lines are slightly inefficient, given that most
        # next_states are also states, for which we already did a forward pass.
        prime_log_pf_all = self.forward_log_prob(valid_next_states)
        valid_log_pf_s_prime_exit=prime_log_pf_all[...,-1]
        preds = all_log_rewards[..., 0] + valid_log_pf_actions + valid_log_pf_s_prime_exit
        ########################
        #   Back_ward passing  #
        ########################
        logpb_all=self.backward_log_prob(valid_next_states)
        valid_log_pb_actions= self.action_prob_gather(logpb_all, actions)
        targets = all_log_rewards[..., 1] + valid_log_pb_actions + valid_log_pf_s_exit
        scores = preds - targets

        if torch.any(torch.isinf(scores)):
            raise ValueError("scores contains inf")
        return scores

