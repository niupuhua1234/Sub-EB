from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional

import torch
from torchtyping import TensorType

from src.gfn.containers.states import States
from src.gfn.envs.env import Env
from src.gfn.samplers import TrajectoriesSampler
# Typing
TensorPmf = TensorType["n_states", float]
class TerminatingStatesDist(ABC):
    """
    Represents an abstract distribution over terminating states.
    """

    @abstractmethod
    def pmf(self,states:States) -> TensorPmf:
        """
        Compute the probability mass function of the distribution.
        """
        pass

class Empirical_Dist(TerminatingStatesDist):
    """
    Represents an empirical distribution over terminating states.
    """
    def __init__(self, env: Env) -> None:
        self.states_to_indices    = env.get_terminating_states_indices
        self.env_n_terminating_states = env.n_terminating_states

    def pmf(self,states) -> TensorPmf:
        assert len(states.batch_shape) == 1, "States should be a linear batch of states"
        states_indices = self.states_to_indices(states)
        counter = Counter(states_indices)
        counter_list = [counter[state_idx] if state_idx in counter else 0
                        for state_idx in range(self.env_n_terminating_states)]
        return torch.tensor(counter_list, dtype=torch.float) / len(states_indices)

class Empirical_Ratio(TerminatingStatesDist):
    """
    Represents an empirical distribution over terminating states.
    """
    def __init__(self,env,
                 sampler:TrajectoriesSampler,
                 B_sampler:TrajectoriesSampler,n_samples=128) -> None:
        self.env       = env
        self.sampler   = sampler
        self.B_sampler = B_sampler
        self.n_samples =n_samples #number of MC samples for each x

    def pmf(self,states) -> TensorPmf:
        repeat_dims =  (1,)*len(states.batch_shape)+(self.n_samples,) + (1,) * len(states.state_shape)
        init           = self.env.States(states.states_tensor.unsqueeze(1).repeat(* repeat_dims))
        B_trajectories = self.B_sampler.sample( states=init.flatten(),fill_value=-1e5)
        index          = ~B_trajectories.is_sink_action
        logpf_trajs   = torch.zeros(B_trajectories.actions.shape,dtype=torch.float,device=states.device)
        logpb_trajs   = torch.zeros(B_trajectories.actions.shape,dtype=torch.float,device=states.device)
        actions        = self.env.bction2action(B_trajectories.states[:-1, ...],B_trajectories.actions)
        logpf_trajs[index] = self.sampler.actions_sampler.get_probs(B_trajectories.states[1:,...][index],actions[index]).log() # s_T-1:s_0
        logpb_trajs[index] = self.B_sampler.actions_sampler.get_probs(B_trajectories.states[:-1][index],B_trajectories.actions[index]).log()# s_T:s_1

        ratios       = (logpf_trajs.sum(0).reshape(-1,self.n_samples)-logpb_trajs.sum(0).reshape(-1,self.n_samples)).exp().mean(-1)
        return  ratios / ratios.sum()

    # def pmf(self,states) -> TensorPmf:
    #     ratio=[]
    #     for state in states:
    #         init           = self.env.States(state.states_tensor.unsqueeze(0).repeat((self.n_samples,1)))
    #         B_trajectories = self.B_sampler.sample(n_trajectories=len(init), states=init,fill_value=-1e5)
    #         actions        = self.env.bction2action(B_trajectories.states[:-1, ...],B_trajectories.actions)
    #         logpf_trajs    = self.sampler.actions_sampler.get_probs(B_trajectories.states[1:,...],actions ).log()
    #         logpb_trajs    = self.B_sampler.actions_sampler.get_probs(B_trajectories.states[:-1],B_trajectories.actions).log()

    #         ratio.append( (logpf_trajs.sum(0)-logpb_trajs.sum(0)).exp().mean())
    #     return torch.tensor(ratio)/torch.tensor(ratio).sum()

    def simple_pmf(self,states) -> TensorPmf:
        assert len(states.batch_shape) == 1, "States should be a linear batch of states"
        states_indices = torch.unique(states.states_tensor,dim=0,return_counts=True)
        return  states_indices[1]/states_indices[1].sum()


