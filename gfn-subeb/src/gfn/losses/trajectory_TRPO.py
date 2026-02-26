from typing import List, Literal, Tuple
import copy
import torch
from torchtyping import TensorType
from src.gfn.containers import Trajectories,States
from src.gfn.losses import TrajectoryRLEval

# Typing
ScoresTensor = TensorType[-1, float]
LossTensor = TensorType[0, float]
LogPTrajectoriesTensor = TensorType["max_length", "n_trajectories", float]


class Trajectory_TRPO(TrajectoryRLEval):

    def __init__(self,*args,delta=1e-2,**kwargs):
        self.delta=delta
        super().__init__(*args,**kwargs)

    def __call__(self, trajectories: Trajectories) -> LossTensor:
        pass

    def update_model(self, trajectories: Trajectories,backward_update=False) -> LossTensor:
        log_pf_traj = self.get_pfs(trajectories)
        log_pb_traj = self.get_pbs(trajectories)
        log_state_fevals = self.get_Vfs(trajectories)

        scores = self.get_scores(trajectories, log_pf_traj.detach(), log_pb_traj.detach())
        advantages = self.advantages_step(trajectories, scores.detach(), - log_state_fevals.detach())

        Z_diff = scores.sum(0).pow(2).mean()
        V_loss = self.evals_step(trajectories,log_pf_traj.detach(),log_pb_traj.detach(),log_state_fevals)
        self.optimizer_step(V_loss, self.V_optimizer)
        if backward_update:
            V2_loss = self.evals_step(trajectories, log_pf_traj.detach(), log_pb_traj, log_state_fevals.detach())
            self.optimizer_step(V2_loss, self.B_optimizer)
        self.optimizer_step(Z_diff, self.A_optimizer)

        A_loss=self.trpo_step(trajectories,advantages,delta=self.delta)
        return A_loss+Z_diff

    # def update_model(self, trajectories: Trajectories,**kwargs) -> LossTensor:
    #     log_pf_trajs = self.get_pfs(trajectories)
    #     log_pb_trajs = self.get_pbs(trajectories)
    #
    #     scores = self.get_scores(trajectories, log_pf_trajs, log_pb_trajs).detach()
    #     Vt = self.get_Vfs(trajectories)
    #     advantages = self.advantages_step(trajectories, scores, Vt.detach())
    #     targets= self.values_step(trajectories, scores, Vt.detach())
    #     Z = self.parametrization.logZ.tensor.exp()
    #
    #     Z_diff = (Z / Z.detach()) * (scores.sum(0).mean())
    #     V_loss = (targets - Vt).pow(2).sum(0).mean()
    #
    #     self.optimizer_step(V_loss, self.V_optimizer)
    #     self.optimizer_step(Z_diff, self.A_optimizer)
    #
    #     A_loss=self.trpo_step(trajectories,advantages)
    #     return A_loss+Z_diff

    def B_trpo_update_model(self, trajectories: Trajectories):
        #TODO: to be verified
        log_pb_trajs = self.get_pbs(trajectories)
        log_pf_traj = self.get_pfs(trajectories)
        scores = (log_pb_trajs[0] - log_pf_traj).detach()
        values = self.get_value(trajectories, backward=True)
        advantages = self.advantages_step(trajectories, scores, values.detach(), unbias=True)
        Qt= self.values_step(trajectories, scores, values.detach(),unbias=True)
        V_loss = (Qt - values).pow(2).sum(0).mean()
        # Kl=self.kl_log_prob(log_pb_traj_all,log_pg_traj_all)
        self.optimizer_step(V_loss, self.VB_optimizer)
        A_loss=self.trpo_step(trajectories,advantages)
        return A_loss  # ,Kl.detach()

    def trpo_step(self, trajectories, advantages, delta=1e-2):
        #delta Should be low (approximately between 0.01 and 0.05) #2e-2 for TRPO-Eval?
        n_traj=trajectories.n_trajectories
        backward=trajectories.is_backward
        if not backward:
            params     =self.parametrization.logit_PF.parameters()
            valid_states, valid_actions, valid_index = self.forward_state_actions(trajectories)
            log_pf_all = self.forward_log_prob(valid_states)
        else:
            params =self.parametrization.logit_PB.parameters()
            valid_states, valid_actions, valid_index = self.backward_state_actions(trajectories)
            log_pf_all = self.backward_log_prob(valid_states)
        #####
        log_pf = self.action_prob_gather(log_pf_all, valid_actions)

        sur_loss=self.surrogate_loss(log_pf,log_pf,advantages[valid_index]).sum(0)/n_traj #
        KL        = self.kl_log_prob(log_pf_all,log_pf_all).sum(0)/n_traj # value is zero, used for Hessian computation at theta_old

        sur_grads = self.flat_grad(sur_loss, params,retain_graph=True)
        kl_grads  = self.flat_grad(KL, params, create_graph=True)       # Create graph, because we will call backward on it (for HVP)
        Hvp       = lambda v:self.flat_grad(kl_grads @ v, params, retain_graph=True)
        search_dir = self.conjugate_gradients(Hvp,sur_grads, n_iter=10) # Hv->inv(H)v

        max_length = torch.sqrt(2 * delta / search_dir.dot(sur_grads))  # vT·inv(H)v,  Hvp(search_dir)=H·inv(H)·v=v
        max_step = max_length * search_dir

        flatten_params = self.flatten(params)

        def line_search(max_step,flatten_params,sur_loss,max_backtracks=20):
            for stepfrac in [.5 ** x for x in range(max_backtracks)]:
                flatten_params_new  = flatten_params+stepfrac * max_step
                self.set_flat_params_to(self.parametrization.logit_PF,flatten_params_new) if not backward else \
                    self.set_flat_params_to(self.parametrization.logit_PB, flatten_params_new)
                log_pf_all_new  = self.forward_log_prob(valid_states) if not backward \
                    else self.backward_log_prob(valid_states)
                log_pf_new = self.action_prob_gather(log_pf_all_new, valid_actions)

                sur_loss_new = self.surrogate_loss(log_pf_new,log_pf,advantages[valid_index]).sum(0)/n_traj
                actual_improve = sur_loss_new-sur_loss
                KL_new = self.kl_log_prob(log_pf_all, log_pf_all_new).mean()
                if actual_improve<0 and KL_new <= delta:
                    return True
                # decreasing objective so improvement should be negative
            return False
        line_search(-max_step,flatten_params,sur_loss)
        return sur_loss

    @staticmethod
    def conjugate_gradients(A, b, n_iter=50, res_tol=1e-5):
        # solve Ax=b  ->torch.linalg.solve(A,b)
        x = torch.zeros_like(b, device=b.device)
        r = b.detach().clone()
        p = b.detach().clone()

        rTr = torch.dot(r, r)

        for i in range(n_iter):
            Ap = A(p)
            alpha = rTr / torch.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            rnTrn = torch.dot(r, r)
            if rnTrn < res_tol:
                print(i)
                break
            p = r + (rnTrn / rTr) * p
            rTr = rnTrn
        return x
    @staticmethod
    def flatten(xs): return  torch.cat([x.view(-1) for x in xs])
    @staticmethod
    def flat_grad(y, x, retain_graph=False, create_graph=False,flat=True):
        # create_graph:created graph of the grad result for higher order gradient computation
        # retain_graph: retain current gradient graph after execution for other gradient computation later.
        if create_graph:  retain_graph = True
        g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        return torch.cat([gg.view(-1) for gg in g]) if flat else g
    @staticmethod
    def set_flat_params_to(model, flat_params):
        n= 0
        for param in model.parameters():
            size = param.numel()
            param.data= flat_params[n:n + size].view(param.shape)
            n += size
