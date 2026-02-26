from typing import List, Literal, Tuple

import torch
from torchtyping import TensorType
from src.gfn.containers import Trajectories
from src.gfn.envs   import Env
from src.gfn.estimators import LogStateFlowEstimator,LogZEstimator,LogitPBEstimator
from src.gfn.losses.base import PFBasedParametrization, TrajectoryDecomposableLoss, Evalmetrization
from src.gfn.samplers import BackwardDiscreteActionsSampler
from src.gfn.containers.states import States
# Typing
ScoresTensor = TensorType[-1, float]
LossTensor = TensorType[0, float]
LogPTrajectoriesTensor = TensorType["max_length", "n_trajectories", float]
FlowTrajectoriesTensor = TensorType["max_length", "n_trajectories", float]

class RLParametrization(PFBasedParametrization):
    r"""
    Exactly the same as DBParametrization
    """
    def __init__(self, logit_PF,logit_PB, logZ: LogZEstimator):
        self.logZ=logZ
        super().__init__(logit_PF,logit_PB)

class TrajectoryRL(TrajectoryDecomposableLoss):
    def __init__(
        self,
        parametrization: RLParametrization,
        optimizer: Tuple[torch.optim.Optimizer,torch.optim.Optimizer | None],
        evalmetrization: Evalmetrization,
        evaloptimizer: Tuple[torch.optim.Optimizer,torch.optim.Optimizer | None],
        env: Env,
        log_reward_clip_min: float = -12,
        lamb:float=0.99,
        lamda: float = 0.99,
    ):
        """
        Args:
             lamda: parameter for bias-variance trade-off
        """
        self.lamb  = lamb
        self.lamda = lamda
        self.env=env
        self.A_optimizer,self.B_optimizer=optimizer
        self.V_optimizer,self.VB_optimizer=evaloptimizer

        super().__init__(parametrization,evalmetrization,fill_value=0.,log_reward_clip_min=log_reward_clip_min)

    def get_scores(
        self, trajectories: Trajectories,
            log_pf_traj:LogPTrajectoriesTensor,
            log_pb_traj:LogPTrajectoriesTensor,
    ) -> ScoresTensor:

        terminal_index=trajectories.is_terminating_action
        log_pb_traj.T[terminal_index.T]=trajectories.log_rewards.clamp_min(self.log_reward_clip_min)\
                                                -self.parametrization.logZ.tensor
        scores= (log_pf_traj-log_pb_traj)
        return  scores

    def get_value(self,trajectories:Trajectories,backward=False):
        flatten_masks = ~trajectories.is_sink_action
        values = torch.full_like(trajectories.actions,dtype=torch.float, fill_value=self.fill_value)
        valid_states = trajectories.states[:-1][flatten_masks]  # remove the dummpy one extra sink states
        values[flatten_masks] = self.evalmetrization.logV(valid_states).squeeze(-1) if not backward \
            else self.evalmetrization.logVB(valid_states).squeeze(-1)
        return values
    def get_Vfs(self, trajectories: Trajectories,backward=False)-> FlowTrajectoriesTensor:
        """Evaluate V function for each action in each trajectory in the batch."""
        if trajectories.is_backward:
            raise AssertionError("Can not compute forward state flow for backward trajectories")
        log_state_evals = torch.full_like(trajectories.actions, fill_value=self.fill_value,dtype=torch.float)
        valid_states,_,valid_mask   =self.forward_state_actions(trajectories)
        log_state_evals[valid_mask] = self.evalmetrization.logV(valid_states).squeeze(-1) if not backward \
            else self.evalmetrization.logVB(valid_states).squeeze(-1)                       #log V(s_0:s_T)
        return log_state_evals

    def get_Vbs(self, trajectories: Trajectories)-> FlowTrajectoriesTensor:
        """Evaluate V function for each action in each trajectory in the batch."""
        if not trajectories.is_backward:
            raise AssertionError("Can not compute backward state evaluation for forward trajectories")
        log_state_evals = torch.full_like(trajectories.actions, fill_value=self.fill_value,dtype=torch.float)
        valid_states,_,valid_mask   = self.backward_state_actions(trajectories)
        log_state_evals[valid_mask] = self.evalmetrization.logVB(valid_states).squeeze(-1) #log V(s_T:s_1)
        return log_state_evals

    def surrogate_loss(self,log_pf, log_qf,advantages):
        """define the loss objective for TRPO"""
        # Its value:    adv
        # Its gradient: adv *▽log p  (= adv* (▽p/p)= ad * {▽exp(logp)/exp(logp)} )
        sur_loss=torch.exp(log_pf - log_qf.detach()).mul(advantages)
        return sur_loss

    def update_model(self, trajectories: Trajectories,**kwargs):
        log_pf_traj= self.get_pfs(trajectories)
        log_pb_traj= self.get_pbs(trajectories)

        scores = self.get_scores(trajectories, log_pf_traj.detach(), log_pb_traj.detach())
        Vt = self.get_Vfs(trajectories)
        advantages = self.advantages_step(trajectories, scores.detach(), Vt.detach())
        targets    = self.values_step(trajectories, scores.detach(), Vt.detach())
        #Z = self.parametrization.logZ.tensor.exp()

        A_loss = self.surrogate_loss(log_pf_traj, log_pf_traj, advantages).sum(0).mean()
        Z_diff = scores.sum(0).pow(2).mean()#(Z / Z.detach()) * (scores.detach().sum(0).mean())
        V_loss = ((targets - Vt).pow(2)).sum(0).mean()

        if isinstance(self.V_optimizer,torch.optim.LBFGS):
            def closure():
                self.V_optimizer.zero_grad()
                val=self.get_Vfs(trajectories)
                V_loss= (targets-val).pow(2).sum(0).mean()
                V_loss.backward()
                return V_loss
            self.V_optimizer.step(closure)
        else:
            self.optimizer_step(V_loss, self.V_optimizer)
        self.optimizer_step(A_loss + Z_diff, self.A_optimizer)
        return A_loss + Z_diff

    def B_update_model(self, trajectories: Trajectories,**kwargs):
        log_pb_traj= self.get_pbs(trajectories)
        log_pf_traj = self.get_pfs(trajectories)

        scores = (log_pb_traj - log_pf_traj).detach()
        Vt = self.get_Vbs(trajectories)
        advantages = self.advantages_step(trajectories, scores, Vt.detach(), unbias=True)
        targets = self.values_step(trajectories, scores, Vt.detach(),unbias=True)
        A_loss = self.surrogate_loss(log_pb_traj, log_pb_traj, advantages).sum(0).mean()
        V_loss = (targets - Vt).pow(2).sum(0).mean()
        # Kl=self.kl_log_prob(log_pb_traj_all,log_pg_traj_all).mean()

        self.optimizer_step(V_loss, self.VB_optimizer)
        self.optimizer_step(A_loss, self.B_optimizer)
        return A_loss  # ,Kl.detach()

    def B_update_model_Emp(self, B_trajectories: Trajectories,**kwargs):
        "TB based optimization for backward trajectory"
        trajectories=B_trajectories.revert_backward_trajectories()
        log_pb_traj= self.get_pbs(trajectories)
        log_pf_traj = self.get_pfs(trajectories)
        scores = self.get_scores(trajectories, log_pf_traj.detach(), log_pb_traj)
        loss=scores.sum(0).pow(2).mean()
        self.optimizer_step(loss, self.B_optimizer)
        return loss  # ,Kl.detach()

    def optimizer_step(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def __call__(self, trajectories: Trajectories):
        log_pf_traj= self.get_pfs(trajectories)
        log_pb_traj= self.get_pbs(trajectories)

        scores = self.get_scores(trajectories, log_pf_traj.detach(), log_pb_traj.detach())
        Vt = self.get_Vfs(trajectories)
        advantages = self.advantages_step(trajectories, scores.detach(), Vt.detach())
        A_loss = self.surrogate_loss(log_pf_traj, log_pf_traj, advantages).sum(0)
        Z = self.parametrization.logZ.tensor.exp()
        Z_diff = (Z / Z.detach()) * (scores.detach().sum(0))#scores.sum(0).pow(2).mean()
        return A_loss + Z_diff

    @staticmethod
    def kl_log_prob(log_prob_q, log_prob_p):
        log_prob_p = log_prob_p.detach()
        kl = (log_prob_p.exp() * (log_prob_p - log_prob_q)).sum(-1)
        return kl
    @staticmethod
    def entropy(log_pf):
        p_log_p = -(log_pf * log_pf.exp()).sum(-1)
        return p_log_p

    def advantages_step(self,trajectories:Trajectories,scores,Vt,unbias=False):
        """
        Returns:
            -advantages:  lamb-biased estimated advantage function
        """
        lamb = 1. if unbias else self.lamb
        masks = ~trajectories.is_sink_action
        Vt_prev = torch.zeros_like(scores[0], dtype=torch.float)
        adv_prev = torch.zeros_like(scores[0], dtype=torch.float)

        deltas = torch.full_like(scores[0], fill_value=0., dtype=torch.float)
        advantages = torch.full_like(scores, fill_value=0., dtype=torch.float)
        for i in reversed(range(scores.size(0))):
            deltas[masks[i]] = scores[i][masks[i]] +  Vt_prev[masks[i]] - Vt[i][masks[i]]
            if torch.any(torch.isnan(deltas)): raise ValueError("NaN in scores")
            Vt_prev = Vt[i]
            ######################################
            adv_prev[masks[i]] = deltas[masks[i]] + lamb * adv_prev[masks[i]]
            advantages[i][masks[i]]= adv_prev[masks[i]]

        advantages[masks] = (advantages[masks] - advantages[masks].mean())
        return advantages

    def values_step(self,trajectories:Trajectories,scores,Vt,unbias=False):
        """
        Returns:
            -tar: lamb-biased estimated :math:`\hat{V}t`
        """
        lamb_V=1. if unbias else self.lamda
        masks = ~trajectories.is_sink_action
        Vt_prev = torch.zeros_like(scores[0], dtype=torch.float)
        tar_prev = torch.zeros_like(scores[0], dtype=torch.float)

        deltas = torch.full_like(scores[0], fill_value=0., dtype=torch.float)
        tar = torch.full_like(scores, fill_value=0., dtype=torch.float)
        for i in reversed(range(scores.size(0))):
            deltas[masks[i]] = scores[i][masks[i]] +  Vt_prev[masks[i]] - Vt[i][masks[i]]
            if torch.any(torch.isnan(deltas)): raise ValueError("NaN in scores")
            Vt_prev = Vt[i]
            ######################################
            tar_prev[masks[i]] = deltas[masks[i]]  + lamb_V* tar_prev[masks[i]]
            tar[i][masks[i]]   = tar_prev[masks[i]]+ Vt[i][masks[i]]

            #tar_prev[masks[i]] = scores[i][masks[i]] + tar_prev[masks[i]] # unbias directly
            #tar[i][masks[i]]   = tar_prev[masks[i]]
        return tar
