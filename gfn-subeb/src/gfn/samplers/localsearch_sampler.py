from typing import List, Optional
import torch
from copy  import deepcopy
from torchtyping import TensorType
from src.gfn.containers import States, Trajectories
from src.gfn.envs import Env
from src.gfn.samplers.actions_samplers import ActionsSampler, BackwardActionsSampler
from src.gfn.samplers.trajectories_sampler import CompleteTrajectoriesSampler
# Typing
StatesTensor = TensorType["n_trajectories", "state_shape", torch.float]
ActionsTensor = TensorType["n_trajectories", torch.long]
LogProbsTensor = TensorType["n_trajectories", torch.float]
DonesTensor = TensorType["n_trajectories", torch.bool]
ForwardMasksTensor = TensorType["n_trajectories", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["n_trajectories", "n_actions - 1", torch.bool]

class LocalsearchSampler(CompleteTrajectoriesSampler):

    def __init__(
            self,
            env: Env,
            actions_sampler: ActionsSampler,
            bctions_sampler: BackwardActionsSampler,
            backratio=0.25, iterations=7
    ):
        """Sample complete trajectories, or completes trajectories from a given batch states, using actions_sampler.

        Args:
            env (Env): Environment to sample trajectories from.
            actions_sampler (ActionsSampler): Sampler of actions.
        """
        super().__init__(env,actions_sampler,bctions_sampler)
        self.backratio= backratio
        self.iterations=iterations

    def sample(self,
               n_trajectories: Optional[int] = None,
               init_states: Optional[States] = None,backward_only=False)->Trajectories:
        n_trajs= n_trajectories//(self.iterations+1)
        assert n_trajectories % (self.iterations + 1) == 0
        trajs   =super().sample(n_trajs)
        last_states=deepcopy(trajs.last_states)
        rewards    = self.env.log_reward(last_states)
        for _ in range(self.iterations):
            B_trajs     =super().sample(init_states=last_states,backward_only=True)
            backtrace_steps    = (B_trajs.when_is_done*(1-self.backratio)).long()
            backtrace_states   = self.env.States(B_trajs.states[backtrace_steps,torch.arange(n_trajs)].states_tensor)
            F_trajs     =super().sample(n_trajs,init_states=backtrace_states)
            update = ( F_trajs .log_rewards>rewards)
            ###################################
            last_states[update] = F_trajs.last_states[update]
            rewards[update]     = F_trajs.log_rewards[update]
            trajs.extend(F_trajs)
        return trajs
