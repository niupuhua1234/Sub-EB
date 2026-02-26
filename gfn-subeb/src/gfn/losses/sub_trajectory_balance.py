from typing import List, Literal, Tuple

import torch
from torchtyping import TensorType

from src.gfn.containers import Trajectories
from src.gfn.estimators import LogStateFlowEstimator
from src.gfn.losses.base import PFBasedParametrization, TrajectoryDecomposableLoss

# Typing
ScoresTensor = TensorType[-1, float]
LossTensor = TensorType[0, float]
LogPTrajectoriesTensor = TensorType["max_length", "n_trajectories", float]
FlowTrajectoriesTensor = TensorType["max_length", "n_trajectories", float]

class SubTBParametrization(PFBasedParametrization):
    r"""
    Exactly the same as DBParametrization
    """
    def __init__(self, logit_PF,logit_PB, logF: LogStateFlowEstimator):
        self.logF=logF
        super().__init__(logit_PF,logit_PB)

class SubTrajectoryBalance(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: SubTBParametrization,
        optimizer: Tuple[torch.optim.Optimizer,torch.optim.Optimizer | None],
        log_reward_clip_min: float = -12,
        weighing: Literal["DB","ModifiedDB","TB",
                          "geometric","equal","geometric_within","equal_within"] = "geometric_within",
        lamb: float = 0.9,
    ):
        """
        Args:
            parametrization: parametrization of the model
            log_reward_clip_min: minimum value of the log-reward. Log-Rewards lower than this value will be clipped to this value. Defaults to -12 (roughly log(1e-5)).
            weighing: how to weigh the different sub-trajectories of each trajectory.
                    - "DB": Considers all one-step transitions of each trajectory in the batch and weighs them equally (regardless of the length of trajectory).
                    Should be equivalent to DetailedBalance loss.
                    - "ModifiedDB": Considers all one-step transitions of each trajectory in the batch and weighs them inversely proportional to the trajectory length.
                            This ensures that the loss is not dominated by long trajectories. Each trajectory contributes equally to the loss.
                    - "TB": Considers only the full trajectory. Should be equivalent to TrajectoryBalance loss.
                    - "equal_within": Each sub-trajectory of each trajectory is weighed equally within the trajectory. Then each trajectory is weighed equally within the batch.
                    - "equal": Each sub-trajectory of each trajectory is weighed equally within the set of all sub-trajectories.
                    - "geometric_within": Each sub-trajectory of each trajectory is weighed proportionally to (lamda ** len(sub_trajectory)), within each trajectory.
                    - "geometric": Each sub-trajectory of each trajectory is weighed proportionally to (lamda ** len(sub_trajectory)), within the set of all sub-trajectories.
            lamb: parameter for geometric weighing
        """
        # Lamda is a discount factor for longer trajectories. The part of the loss
        # corresponding to sub-trajectories of length i is multiplied by lamda^i
        # where an edge is of length 1. As lamda approaches 1, each loss becomes equally weighted.
        super().__init__(parametrization,log_reward_clip_min=log_reward_clip_min)
        self.weighing = weighing
        self.lamb = lamb
        self.A_optimizer,self.B_optimizer=optimizer

    def update_model(self,trajectories  : Trajectories,  backward_update=True):
        log_pf_traj          = self.get_pfs(trajectories)
        log_pb_traj          = self.get_pbs(trajectories)
        log_state_fevals     = self.get_Fs(trajectories)
        loss=self.loss_step(trajectories, log_pf_traj, log_pb_traj.detach(), log_state_fevals)
        self.optimizer_step(loss,self.A_optimizer)

        if backward_update:
            loss2= self.loss_step(trajectories, log_pf_traj.detach(), log_pb_traj, log_state_fevals.detach())
            self.optimizer_step(loss2, self.B_optimizer)
        return loss

    def B_update_model(self,B_trajectories: Trajectories, forward_update=True):
        trajectories=B_trajectories.revert_backward_trajectories() \
            if B_trajectories.is_backward else B_trajectories
        log_pf_traj          = self.get_pfs(trajectories)
        log_pb_traj          = self.get_pbs(trajectories)
        log_state_fevals     = self.get_Fs(trajectories)
        loss                 =self.loss_step(trajectories, log_pf_traj.detach(), log_pb_traj, log_state_fevals.detach())
        self.optimizer_step(loss,self.B_optimizer)

        if forward_update:
            loss2 = self.loss_step(trajectories, log_pf_traj, log_pb_traj.detach(), log_state_fevals)
            self.optimizer_step(loss2, self.A_optimizer)
        return loss

    def get_Fs(self, trajectories: Trajectories)-> FlowTrajectoriesTensor:
        """Evaluate log_F for each action in each trajectory in the batch."""
        log_state_flows = torch.full_like(trajectories.actions, fill_value=self.fill_value,dtype=torch.float)
        valid_states,_,valid_mask=self.forward_state_actions(trajectories)
        log_state_flows[valid_mask] = self.parametrization.logF(valid_states).squeeze(-1) #log F(s_0:s_{T})
        return log_state_flows

    def __call__(self, trajectories: Trajectories):
        pass

    def loss_step(self, trajectories: Trajectories,log_pf_trajs,log_pb_trajs,log_state_flows) -> LossTensor:
        score_list,flattening_mask_list= self.get_scores_all(trajectories,log_pf_trajs,log_pb_trajs,log_state_flows)
        loss                           = self.get_losses_all(trajectories,score_list,flattening_mask_list,self.lamb,self.weighing)
        return loss
