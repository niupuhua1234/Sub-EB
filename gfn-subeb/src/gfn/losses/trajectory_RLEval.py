from typing import List, Literal, Tuple
import copy
import torch
from torchtyping import TensorType
from src.gfn.containers import Trajectories,States
from src.gfn.losses import TrajectoryRL
# Typing
ScoresTensor = TensorType[-1, float]
LossTensor = TensorType[0, float]
LogPTrajectoriesTensor = TensorType["max_length", "n_trajectories", float]

class TrajectoryRLEval(TrajectoryRL):
    def __init__(self,
                 *args,
                 weighing: Literal["DB","ModifiedDB","TB","geometric","equal","geometric_within","equal_within"]='geometric_within',
                 no_Z=False,
                 **kwargs,
                 ):
        """
        Args:
             lamda: parameter for bias-variance trade-off
        """
        self.weighing=weighing
        self.no_Z=no_Z
        super().__init__(*args, **kwargs)


    def __call__(self, trajectories: Trajectories) -> LossTensor:
        log_pf_traj     = self.get_pfs(trajectories)
        log_pb_traj     = self.get_pbs(trajectories)
        log_state_fevals= self.get_Vfs(trajectories)

        scores    = self.get_scores(trajectories, log_pf_traj.detach(), log_pb_traj.detach())
        advantages= self.advantages_step(trajectories, scores.detach(), - log_state_fevals.detach())
        # note for evalution balance, we are estimating -V for RL methods !!!!!!
        #
        A_loss = self.surrogate_loss(log_pf_traj, log_pf_traj, advantages).sum(0)
        Z_diff = scores.sum(0).pow(2)
        return A_loss + Z_diff

    def evals_step(self, trajectories: Trajectories, log_pf_trajs, log_pb_trajs,log_state_fevals):
        eval_list,flattening_mask_list= self.get_scores_all(trajectories,log_pf_trajs,log_pb_trajs, log_state_fevals)
        loss                        =   self.get_losses_all(trajectories,eval_list,flattening_mask_list,self.lamda,self.weighing)
        return loss

    def update_model(self, trajectories: Trajectories,backward_update=False):
        log_pf_traj     = self.get_pfs(trajectories)
        log_pb_traj     = self.get_pbs(trajectories)
        log_state_fevals= self.get_Vfs(trajectories)

        scores    = self.get_scores(trajectories, log_pf_traj.detach(), log_pb_traj.detach())
        advantages= self.advantages_step(trajectories, scores.detach(), - log_state_fevals.detach())
        # note for evalution balance, we are estimating -V for RL methods !!!!!!
        #
        A_loss = self.surrogate_loss(log_pf_traj, log_pf_traj, advantages).sum(0).mean()
        Z_diff = scores.sum(0).pow(2).mean()
        V_loss = self.evals_step(trajectories,log_pf_traj.detach(),log_pb_traj.detach(),log_state_fevals)
        self.optimizer_step(V_loss, self.V_optimizer)
        if backward_update:
            V2_loss = self.evals_step(trajectories, log_pf_traj.detach(), log_pb_traj, log_state_fevals.detach())
            self.optimizer_step(V2_loss, self.B_optimizer)

        self.optimizer_step(A_loss + Z_diff, self.A_optimizer)

        return A_loss + Z_diff

    def B_update_model(self, B_trajectories: Trajectories,forward_update=False):
        B_log_pf_traj     = self.get_pfs(B_trajectories)
        B_log_pb_traj     = self.get_pbs(B_trajectories)
        Vt        = -    self.get_Vbs(B_trajectories).detach() # note for evalution balance, we are estimating -V for RL methods !!!!!!
        B_scores    = (B_log_pb_traj - B_log_pf_traj).detach()
        advantages= self.advantages_step(B_trajectories, B_scores, Vt)
        A_loss = self.surrogate_loss(B_log_pb_traj, B_log_pb_traj, advantages).sum(0).mean()
        ##################
        trajectories = B_trajectories.revert_backward_trajectories()
        log_pf_traj     = self.get_pfs(trajectories)
        log_pb_traj     = self.get_pbs(trajectories)
        log_state_evals = self.get_Vfs(trajectories,backward=True)

        V_loss = self.evals_step(trajectories,log_pf_traj.detach(),log_pb_traj.detach(),log_state_evals)
        self.optimizer_step(V_loss, self.VB_optimizer)
        if forward_update:
            scores    = self.get_scores(trajectories, log_pf_traj.detach(), log_pb_traj.detach())
            Z_diff    = scores.sum(0).pow(2).mean() if not self.no_Z else 0.
            V2_loss = self.evals_step(trajectories, log_pf_traj, log_pb_traj.detach(), log_state_evals.detach())
            self.optimizer_step(V2_loss+Z_diff, self.A_optimizer)

        self.optimizer_step(A_loss, self.B_optimizer)
        return A_loss

    def B_MLE(self, trajectories: Trajectories):
        log_pb_traj     = self.get_pbs(trajectories)
        MLE    = torch.sum(-log_pb_traj,0).mean().clamp(max=1e4)
        self.optimizer_step(MLE, self.B_optimizer)
        return MLE