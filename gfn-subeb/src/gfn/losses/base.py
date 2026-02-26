import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple,List
import torch
from torchtyping import TensorType

from src.gfn.containers.states import States
from src.gfn.containers.trajectories import Trajectories
from src.gfn.containers.transitions import Transitions
from src.gfn.estimators import LogitPBEstimator, LogitPFEstimator,LogEdgeFlowEstimator,LogStateFlowEstimator
LogPTrajectoriesTensor = TensorType["max_length", "n_trajectories", float]
FlowTrajectoriesTensor = TensorType["max_length", "n_trajectories", float]
ScoresTensor = TensorType[-1, float]
LossTensor = TensorType[0, float]
####################
class Parametrization(ABC):
    """
    Abstract Base Class for Flow Parametrizations,
    as defined in Sec. 3 of GFlowNets Foundations.
    All attributes should be estimators, and should either have a GFNModule or attribute called `module`,
    or torch.Tensor attribute called `tensor` with requires_grad=True.
    """
    @property
    def parameters(self) -> dict:
        """
        Return a dictionary of all parameters of the parametrization.
        Note that there might be duplicate parameters (e.g. when two NNs share parameters),
        in which case the optimizer should take as input set(self.parameters.values()).
        """
        # TODO: use parameters of the fields instead, loop through them here
        parameters_dict = {}
        for name,estimator in self.__dict__.items():
            parameters_dict.update({name:estimator.parameters()})
        return parameters_dict

    def save_state_dict(self, path: str,index:str):
        for name, estimator in self.__dict__.items():
            torch.save(estimator.named_parameters(), os.path.join(path, index+name + ".pt"))

    def load_state_dict(self, path: str,index:str):
        for name, estimator in self.__dict__.items():
            estimator.load_state_dict(torch.load(os.path.join(path, index+name + ".pt")))

class FParametrization(Parametrization):
    def __init__(self, logF: LogEdgeFlowEstimator):
        self.logF=logF

class PFBasedParametrization(Parametrization, ABC):
    r"Base class for parametrizations that explicitly used :math:`P_F`"
    def __init__(self, logit_PF: LogitPFEstimator,logit_PB: LogitPBEstimator):
        self.logit_PF=logit_PF
        self.logit_PB=logit_PB

class Evalmetrization(Parametrization, ABC):
    r"Base class for parametrizations that explicitly used :math:`P_F`"
    def __init__(self, logV: LogStateFlowEstimator|LogEdgeFlowEstimator,
                 logVB: LogStateFlowEstimator|LogEdgeFlowEstimator):
        self.logV=logV
        self.logVB=logVB
####################################
# loss objects                     #
# ##################################

class Loss(ABC):
    "Abstract Base Class for all GFN Losses"
    def __init__(self, parametrization, evalmetrization=None):
        self.parametrization=parametrization
        self.evalmetrization=evalmetrization
    @abstractmethod
    def __call__(self, *args, **kwargs) -> TensorType[0, float]:
        pass

class Sub_TrajectoryDecomposableLoss(Loss,ABC):
    """
    Args:
        fill_value (float, optional):  LogP Value to use for invalid states (i.e. s_f that is added to shorter trajectories). Defaults to 0.0.
                                       Here we used 0.0 instead of inf_value to ensure stability.
        inf_value (float, optional):   LogP Value to use for zero probability.                      Defaults to -1e5 ( or -float('inf')).
        temperature (float, optional): Temperature to use for the softmax(correspond to how the actions_sampler evaluates each action.). Defaults to 1.0.
    """
    def __init__(self, parametrization: PFBasedParametrization,
                 evalmetrization: Evalmetrization=None,
                 fill_value=0.0,## only inf &0 can avoid masked log_value increase after summation of cumlog_prob
                 temperature=1.0,
                 inf_value=-1e5,# not -inf for graidents stability
                 log_reward_clip_min=-12,
                 ):
        self.fill_value =fill_value
        self.temperature=temperature
        self.inf_value:float=inf_value
        self.log_reward_clip_min= log_reward_clip_min
        super().__init__(parametrization,evalmetrization)

    @abstractmethod
    def __call__(self, states_tuple: Tuple[States, States]) -> TensorType[0, float]:
        pass
    @staticmethod
    def action_prob_gather(log_ps,actions):
        return torch.gather(log_ps, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

    def forward_log_prob(self,states: States):
        logits=self.parametrization.logit_PF(states)
        if torch.any(torch.all(torch.isnan(logits), 1)):
            raise ValueError("NaNs in estimator")
        logits[~states.forward_masks] = self.inf_value            # note we use self.inf_value rathern self.fill_value
        log_all = (logits/ self.temperature).log_softmax(dim=-1)  # By log_softmax inf_value in logits can be recovered
        return log_all

    def backward_log_prob(self,states: States):
        logits=self.parametrization.logit_PB(states)
        if torch.any(torch.all(torch.isnan(logits), 1)): raise ValueError("NaNs in estimator")
        logits[~states.backward_masks] = self.inf_value
        log_all = logits.log_softmax(dim=-1)
        return log_all

    @staticmethod
    def forward_state_actions(trajectories: Trajectories):
        """
        1. compute forward_prob for forward trajectory.
        Forward trajectory:                s0   ->    s1   ->..... ->   sT-1  ->s_T   (->sf)
                                           a0   ->    a1   ->....  ->   aT-1  ->a_T
        compute forward probability for: π|s0   ->  π|s1   ->..... -> π|sT-1  ->π|s_T
        2. compute forward_prob for backward trajectory. (used for backward guided policy)
        Backward trajectory:                          sT    ->    sT-1 -> ....  ->  s1 (-> s0)
                                                      aT-1   ->    aT-2 ->   ....->  a0
        compute forward probability for state: (ST)-> π|sT-1 ->   π|sT-2->...... ->π|s0 (错一格 和a对齐)

        """
        if not(trajectories.is_backward):
            valid_index = ~trajectories.is_sink_action            # filtered padding s_f
            valid_states= trajectories.states[:-1][valid_index]   #s0->sf: [:-1] states traj is one bigger than actions traj by an s_f in forward samples
            valid_actions = trajectories.actions[valid_index]
        else:
            valid_index = ~trajectories.is_sink_action
            valid_states= trajectories.states[1:][valid_index]    #sT<-s0: [1:] states traj is one bigger than actions traj by an s_T in  backward samples
            valid_actions=trajectories.env.bction2action(trajectories.states[:-1][valid_index], trajectories.actions[valid_index])
        if valid_states.batch_shape != tuple(valid_actions.shape):
            raise AssertionError("Something wrong happening with log_pf evaluations")
        return valid_states,valid_actions,valid_index
    @staticmethod
    def backward_state_actions(trajectories: Trajectories):
        """
        1. compute backward_prob for forward trajectory.
        Forward trajectory:                       s0   ->    s1   ->..... ->  sT-1  (->sT ->sf )
                                                  a0   ->    a1   ->....  ->  aT-1  (->aT)
        compute back probability for: (s0)  ->  π|s1   ->.....  ->π|sT-1  ->  π|sT             (s错一格,和a 对齐)
        2. compute backward_prob for backward trajectory. (used for backward guided policy)
        Backward trajectory:            sT  ->    sT-1 ->...... ->  s1   (->s_0 )
                                       (bT  ->    bt-1 -> ......->  b1)
                                      aT-1  ->    aT-2 ->.......->  a0
        compute back probability for: π|sT  ->  π|sT-1 ->...... ->π|s1
        """
        if not (trajectories.is_backward):
            inter_index=~trajectories.is_sink_action & ~trajectories.is_terminating_action # filter padding s_f and s_T
            non_init_valid_states  =  trajectories.states[1:][inter_index]                 # [1:] align with action traj and select s1->s_T instead of s0->sT-1
            non_exit_valid_actions = trajectories.env.action2bction( trajectories.states[:-1][inter_index],
                                                                     trajectories.actions[inter_index])
        else:
            inter_index = ~trajectories.is_sink_action                     # filter padding s_f( dummy states after reaching s0)
            non_init_valid_states = trajectories.states[:-1][inter_index]  #sT->s0 [:-1] state traj is one bigger than actions traj by an s_0
            non_exit_valid_actions = trajectories.actions[inter_index]
        return non_init_valid_states,non_exit_valid_actions,inter_index

    def get_pfs(
        self,
        trajectories: Trajectories,is_all=False) -> LogPTrajectoriesTensor| LogPTrajectoriesTensor:
        """Evaluate log_pf for each action in each trajectory in the batch.
        Args:
            trajectories (Trajectories): Trajectories to evaluate.
            fill_value   (float)       : Values used for invalid states (sink state usually)
        Returns:
            Tuple[LogPTrajectoriesTensor | None, LogPTrajectoriesTensor]: A tuple of float tensors of shape (max_length, n_trajectories)
             containing the log_pf and log_pb for each action in each trajectory.
        """
        valid_states, valid_actions,valid_index=self.forward_state_actions(trajectories)
        valid_log_pf_all=self.forward_log_prob(valid_states)
        if  not is_all:
            valid_log_pf_actions = self.action_prob_gather(valid_log_pf_all, valid_actions)
            # assert torch.all((trajectories.log_probs[trajectories.actions != -1] - valid_log_pf_actions).abs() < 1e-3) if on policy
            log_pf_trajectories = torch.full_like(trajectories.actions, fill_value=self.fill_value, dtype=torch.float)
            log_pf_trajectories[valid_index] = valid_log_pf_actions
            return log_pf_trajectories
        else:
            log_pf_trajectories_all = torch.full_like(trajectories.states[:-1].forward_masks, fill_value=self.fill_value, dtype=torch.float)
            log_pf_trajectories_all[valid_index,:] = valid_log_pf_all
            return log_pf_trajectories_all

    def get_pbs(
            self,
            trajectories: Trajectories,is_all=False) -> LogPTrajectoriesTensor |LogPTrajectoriesTensor:
        """Evaluate log_pb for each action in each trajectory in the batch."""
        non_init_valid_states,non_exit_valid_actions,inter_index=self.backward_state_actions(trajectories)
        valid_log_pb_all=self.backward_log_prob(non_init_valid_states)

        if  not is_all:
            valid_log_pb_actions = self.action_prob_gather(valid_log_pb_all, non_exit_valid_actions)
            # torch.all((trajectories.log_probs[trajectories.actions != -1][valid_actions != trajectories.env.n_actions - 1] - valid_log_pb_actions).abs() <= 1e-4)
            log_pb_trajectories = torch.full_like(trajectories.actions, fill_value=self.fill_value, dtype=torch.float)
            log_pb_trajectories[inter_index] = valid_log_pb_actions
            return log_pb_trajectories
        else:
            log_pb_trajectories_all = torch.full_like(trajectories.states[:-1].backward_masks, fill_value=self.fill_value, dtype=torch.float)
            log_pb_trajectories_all[inter_index] = valid_log_pb_all
            return log_pb_trajectories_all

    def optimizer_step(self, loss, optimizer,):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class StateDecomposableLoss(Sub_TrajectoryDecomposableLoss,ABC):
    def __init__(self, parametrization: FParametrization,
                 fill_value=0.0,
                 inf_value=-1e5):
        self.fill_value = fill_value
        self.inf_value:float  =inf_value
        Loss.__init__(self,parametrization)
    @abstractmethod
    def __call__(self, states_tuple: Tuple[States, States]) -> TensorType[0, float]:
        pass
    @staticmethod
    def action_flow_gather(log_F,actions):
        return torch.gather(log_F, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    def forward_log_prob(self,states: States):
        logits=self.parametrization.logF(states)
        if torch.any(torch.all(torch.isnan(logits), 1)):
            raise ValueError("NaNs in estimator")
        logits[~states.forward_masks] = self.inf_value            # note we use self.inf_value rathern self.fill_value
        log_all = logits.log_softmax(dim=-1)  # By log_softmax inf_value in logits can be recovered
        return log_all
class EdgeDecomposableLoss(Sub_TrajectoryDecomposableLoss,ABC):
    @abstractmethod
    def __call__(self, edges: Transitions) -> TensorType[0, float]:
        pass

class TrajectoryDecomposableLoss(Sub_TrajectoryDecomposableLoss,ABC):
    @abstractmethod
    def __call__(self, trajectories: Trajectories) -> TensorType[0, float]:
        pass

    def cumulative_logprobs(
        self,
        trajectories: Trajectories,
        log_p_trajectories: LogPTrajectoriesTensor,
    ):
        """
        Args:
            trajectories: trajectories
            log_p_trajectories: log probabilities of each transition in each trajectory
        Return:
             cumulative max_length+1 sums of log probabilities of each trajectory.
        e.g. forward prob of forward traj:
            Σ  log π|st  [ 0,        Σ(0:1)         ,....., Σ(0:3)     ,....., Σ(0:T) ,       Σ(0:f)   ].
                           0, logp(τ_{0:1}|s_0),....., logp(τ_{0:3}|0) ,....., logp(τ_{0:T}|0),logp(τ_{0:f}|0)
            backward prob of forward traj:
            Σ  log π|st  [ 0,       Σ(0:1)     , ....,  Σ(0:3)         ,....., Σ(0:T),         ____________].
                           0,  logp(τ_{0:1}|s_1),....., logp(τ_{0:3}|3),....., logp(τ_{0:T}|T), _____________
        """
        return torch.cat((torch.zeros(1, trajectories.n_trajectories, device=log_p_trajectories.device),
                          log_p_trajectories.cumsum(dim=0),),dim=0)

    def get_scores_all(
        self, trajectories: Trajectories,
            log_pf_trajectories: LogPTrajectoriesTensor,
            log_pb_trajectories: LogPTrajectoriesTensor,
            log_state_flows    : FlowTrajectoriesTensor
    ) -> Tuple[List[ScoresTensor], List[ScoresTensor]]:
        """
            Returns:
                scores : A list of tensors for k in [1, ...,traj.max_length], each of which representing the evaluations of all sub-trajectories of length k (k+1 states, k transitions).
                         The score of a sub-trajectory tau is  :math:`log P_F(tau_i:j|i) + log V(tau_i) - log P_B(tau_i:j|j) - log V(tau_j)`.
                         The shape of the k-th tensor is ( traj.max_length-k+1, traj.n_trajectories).
                flattening_masks: A list of tensors representing what should be masked out in each element of the first list, given that not all sub-trajectories
                                  of length k exist for each trajectory. The entries of those tensors are True if the corresponding sub-trajectory does not exist.
        """
        log_pf_trajectories_cum = self.cumulative_logprobs(trajectories, log_pf_trajectories)
        log_pb_trajectories_cum = self.cumulative_logprobs(trajectories, log_pb_trajectories)
        log_Z=self.parametrization.logZ.tensor.detach() if hasattr(self.parametrization,'logZ') else 0.#trajectories.env.logZ_est

        valid_mask = ~trajectories.is_sink_action
        terminal_mask =  trajectories.is_terminating_action
        inter_mask = valid_mask & ~terminal_mask

        flattening_masks = []
        scores = []
        #   For traj.max_length=n, the flow difference of sub-traj.max_length=k∈(1,n) that start from (0,...,n-k)
        for j in range(trajectories.max_length):
            current_log_state_flows = (log_state_flows if j == 0 else log_state_flows[: -j])  #logF (s0:sn-k)
            preds = (log_pf_trajectories_cum[j+1:]-log_pf_trajectories_cum[:-j-1]+ current_log_state_flows)
            #cum_logp[k,... ] :           Σ(0:k),   Σ(0:k+1)         ,....,Σ(0:sT)     , Σ(0:sf)
            #cum_logp[...,-k] :           Σ(0:0),   Σ(0:1  )         ,....,Σ(0:sT-k)   , Σ(0:sn-k)
            # A-B             :  logp(τ_{0:k}|s0), logp(τ_{1:k+1}|s1),....,            , logp(τ_{n-k:n}|s_{n-k}),

            targets = torch.full_like(preds, fill_value=self.fill_value)
            targets.T[terminal_mask[j :].T] = (trajectories.log_rewards[trajectories.when_is_done > j]
                                               .clamp_min(self.log_reward_clip_min)-log_Z)#
            if j > 0: targets[terminal_mask[j :]] += (log_pb_trajectories_cum[j :] -log_pb_trajectories_cum[: -j])[:-1][terminal_mask[j :]]
            # logR(sT→sf)+ logp(τ_{T-k:T}|sT),        logp(τ_{T-k:T}|sT)= Σ(0: sT)  - Σ(0: sT-k)
            backward_log_state_flows =  log_state_flows[j+1:][valid_mask[j+1:]]   #log F(sk:sT)
            targets[inter_mask[j :]] = (log_pb_trajectories_cum[j+1:] -log_pb_trajectories_cum[:-j-1])[inter_mask[j:]] + backward_log_state_flows
            #cum[k:]                       Σ(0: k),  Σ(0: k+1)          ,...,Σ(0: sT)
            #cum[:-k]                      Σ(0: 0),  Σ(0:1 )            ,...,Σ(0: sT-k)
            #A-B           : logp(τ_{0:k}|s_k),logp(τ_{0:k+1}|s_k+1),...,logp(τ_{T-k:T}|sT)

            flattening_mask = trajectories.when_is_done.lt(torch.arange(j+1,trajectories.max_length + 1,device=trajectories.when_is_done.device,).unsqueeze(-1))
            flat_preds = preds[~flattening_mask]   # masking scores for trajs with length <k keep trajs ∈[k, n]
            flat_targets = targets[~flattening_mask]

            if torch.any(torch.isnan(flat_preds)): raise ValueError("NaN in preds")
            if torch.any(torch.isnan(flat_targets)):raise ValueError("NaN in targets")

            flattening_masks.append(flattening_mask)
            scores.append(preds - targets)

        return scores,flattening_masks


    def get_losses_all(
            self,
            trajectories:Trajectories,
            score_list:List[ScoresTensor],
            flattening_mask_list:List[ScoresTensor],
            lamb,weighing
    )->LossTensor:
        flattening_mask = torch.cat(flattening_mask_list)
        all_scores = torch.cat(score_list, 0)  # (n_rows, n_trajectories)

        max_length = trajectories.max_length
        lengths    = trajectories.when_is_done
        n_rows     = int(max_length * (1 + max_length) / 2)  # maximum of n_subs
        n_subs     = lengths * (lengths + 1) / 2  # the number of sub-trajs that each traj contains
        device = trajectories.actions.device

        if weighing == "equal_within":
            # weight of sub_traj_i,j    =1/ the number of sub-traj that the traj_i contains
            # weight of traj_i in batch = Σj  1/ the number of sub-traj_i,j that traj_i contains =1
            contributions = 1.0 / n_subs
            contributions = contributions / len(trajectories)
            contributions = contributions.repeat(n_rows, 1)
        elif weighing == "equal":  # weight of each sub-traj = 1/ total number of sub-traj
            n_sub_trajectories = int(n_subs.sum().item())
            contributions = torch.ones(n_rows, len(trajectories),device=device) / n_sub_trajectories
        elif weighing == "geometric_within":
            #  weight of sub_traj_i,j= λ^len(sub_traj_i,j)
            contributions = (lamb ** torch.arange(max_length,device=device).double()).float()
            contributions = contributions.unsqueeze(-1).repeat(1, len(trajectories))
            # repeat λ^0 for n times,  λ^1 for n-1 times,..... λ^(n-1) for 1 times
            contributions = contributions.repeat_interleave(torch.arange(max_length, 0, -1,device=device), dim=0, output_size=n_rows)
            per_trajectory_denominator = (contributions * (~flattening_mask)).sum(0)

            contributions = contributions / per_trajectory_denominator
            contributions = contributions / len(trajectories)
        elif weighing == "geometric_within_unnormal":
            #  weight of sub_traj_i,j= λ^len(sub_traj_i,j)
            contributions = (lamb ** torch.arange(max_length,device=device).double()).float()
            contributions = contributions.unsqueeze(-1).repeat(1, len(trajectories))
            # repeat λ^0 for n times,  λ^1 for n-1 times,..... λ^(n-1) for 1 times
            contributions = contributions.repeat_interleave(torch.arange(max_length, 0, -1,device=device), dim=0, output_size=n_rows)

            losses=torch.full_like(all_scores,fill_value=0.,dtype=torch.float)
            losses[~flattening_mask]=contributions[~flattening_mask]*all_scores[~flattening_mask].pow(2)
            return losses.sum(0).mean()

        elif weighing == "geometric_withinf":
            #  weight of sub_traj_i,j= λ^len(sub_traj_i,j)
            contributions = (lamb ** (torch.arange(max_length,0,-1,device=device)-1).double()).float()
            contributions = contributions.unsqueeze(-1).repeat(1, len(trajectories))
            # repeat λ^0 for n times,  λ^1 for n-1 times,..... λ^(n-1) for 1 times
            contributions = contributions.repeat_interleave(torch.arange(max_length, 0, -1,device=device), dim=0, output_size=n_rows)
            per_trajectory_denominator=(contributions*(~flattening_mask)).sum(0)

            contributions = contributions / per_trajectory_denominator
            contributions = contributions / len(trajectories)
        elif weighing == "geometric":
            # The k-th entry represents the mean of all losses of sub-trajectories of length k
            per_length_losses = torch.stack([evals[~flattening_mask].pow(2).mean()
                                             for evals, flattening_mask in zip(score_list, flattening_mask_list)])
            weights = ((1 - lamb) / (1 - lamb ** max_length) *
                       (lamb ** torch.arange(max_length, device=per_length_losses.device)))
            assert (weights.sum() - 1.0).abs() < 1e-5, f"{weights.sum()}"
            return (per_length_losses * weights).sum()
        elif weighing == "TB":
            indices = (max_length * (lengths - 1) - (lengths - 2 + 1) * (lengths - 2) / 2).long()
            # 等差数列 减去多数的index  首项 1 末项 traj_lenth-2   一共traj_lenth-2 个
            return all_scores[indices, torch.arange(len(trajectories))].pow(2).mean()
        elif weighing == "DB":
            # Longer trajectories contribute more to the loss
            return score_list[0][~flattening_mask_list[0]].pow(2).mean()  # only consider trajtories with length 1

        elif weighing == "ModifiedDB":
            # The following tensor represents the inverse of how many transitions there are in each trajectory
            contributions = 1.0 / lengths
            contributions = contributions / len(trajectories)
            contributions = contributions.repeat(max_length, 1)
            contributions = torch.cat(
                (contributions, torch.zeros((n_rows - max_length, len(trajectories)), device=contributions.device)), 0)
        else:
            raise ValueError(f"Unknown weighing method {weighing}")

        flat_contributions = contributions[~flattening_mask]
        assert (flat_contributions.sum() - 1.0).abs() < 1e-5, f"{flat_contributions.sum()}"
        losses = flat_contributions * all_scores[~flattening_mask].pow(2)
        return losses.sum()

