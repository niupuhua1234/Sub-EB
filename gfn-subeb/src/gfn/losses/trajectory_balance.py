from dataclasses import dataclass
from typing import Tuple

import torch
from torchtyping import TensorType
from torch.nn.functional import huber_loss
from src.gfn.containers import Trajectories
from src.gfn.estimators import LogZEstimator
from src.gfn.losses.base import PFBasedParametrization, TrajectoryDecomposableLoss
# from src.gfn.samplers.actions_samplers import (
#     BackwardDiscreteActionsSampler,
#     DiscreteActionsSampler,
# )

# Typing
ScoresTensor = TensorType["n_trajectories", float]
LossTensor = TensorType[0, float]
LogPTrajectoriesTensor = TensorType["max_length", "n_trajectories", float]

class TBParametrization(PFBasedParametrization):
    r"""
    :math:`\mathcal{O}_{PFZ} = \mathcal{O}_1 \times \mathcal{O}_2 \times \mathcal{O}_3`, where
    :math:`\mathcal{O}_1 = \mathbb{R}` represents the possible values for logZ,
    and :math:`\mathcal{O}_2` is the set of forward probability functions consistent with the DAG.
    :math:`\mathcal{O}_3` is the set of backward probability functions consistent with the DAG, or a singleton
    thereof, if self.logit_PB is a fixed LogitPBEstimator.
    Useful for the Trajectory Balance Loss.
    """
    def __init__(self, logit_PF,logit_PB, logZ: LogZEstimator):
        self.logZ=logZ
        super().__init__(logit_PF,logit_PB)

class TrajectoryBalance(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: TBParametrization,
        optimizer:torch.optim.Optimizer,
        log_reward_clip_min: float = -12,
    ):
        """Loss object to evaluate the TB loss on a batch of trajectories.

        Args:
            log_reward_clip_min (float, optional): minimal value to clamp the reward to. Defaults to -12 (roughly log(1e-5)).
            on_policy (bool, optional): If True, the log probs stored in the trajectories are used. Defaults to False.

        Forward     P(·|s0)→  ...  →P(sT|sn-1) → P(sf|sT)
        -------------------------------------------------
        Backward    P(s0|s1)←  ... ←P(sn-1|sT) ← P(sT|sf)
                      =1                           R(sT)/Z
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Forward     P(·|s0)→  ...  →P(sT|sn-1) → P(sf|sT)                if  not all states are terminatiing states
                                                    =1
        -------------------------------------------------
        Backward    P(s0|s1)←   ...←P(sn-1|sT) ←P(sT|sf)
                      =1                          R(sT)/Z
        """
        super().__init__(parametrization,log_reward_clip_min=log_reward_clip_min)
        self.optimizer=optimizer

    def get_scores(self, trajectories: Trajectories,
                   log_pf_trajectories:LogPTrajectoriesTensor,
                   log_pb_trajectories:LogPTrajectoriesTensor) -> ScoresTensor:
        terminal_index=trajectories.is_terminating_action
        # TODO:通过bool matrix 索引, 取值的顺序是 按 matrix 的行 进行的
        # TODO:因此当 （time, batch),  bool 取到的值 不是按 batch 顺序 取的而是 time 顺序
        pred  =log_pf_trajectories
        pred[0,:] +=self.parametrization.logZ.tensor
        target=log_pb_trajectories
        target.T[terminal_index.T]+=trajectories.log_rewards.clamp_min(self.log_reward_clip_min)
        scores= (pred-target).sum(0)
        return scores

    def update_model(self,trajectories: Trajectories,**kwargs):
        loss=self.__call__(trajectories)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def __call__(self, trajectories: Trajectories,reduce=True) -> LossTensor:
        log_pf_trajectories = self.get_pfs(trajectories)
        log_pb_trajectories = self.get_pbs(trajectories)
        scores= self.get_scores(trajectories, log_pf_trajectories ,log_pb_trajectories)
        loss=scores.pow(2).mean() if reduce else scores.pow(2)
        #loss=huber_loss(scores)
        if torch.isnan(loss).any():raise ValueError("loss is nan")
        return loss
