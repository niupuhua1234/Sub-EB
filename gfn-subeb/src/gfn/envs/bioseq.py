from abc import ABC, abstractmethod
from typing import ClassVar, Literal, Tuple, List

import pickle
import numpy as np
import torch
from gymnasium.spaces import Discrete
from torchtyping import TensorType
from src.gfn.containers.states import States
from src.gfn.envs.env import Env
from src.gfn.envs.preprocessors import KHotPreprocessor
import os
# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]
BatchTensor = TensorType["batch_shape"]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
def nbase2dec(n,b, length):
    #n:n_base b:bits
    canonical_base = n ** torch.arange(length - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(canonical_base * b, -1)
class Oracle(ABC):
    def __init__(self, nbase,ndim,oracle_path,mode_path=None,reward_exp=3,reward_max=10.0,reward_min=1e-3,name="TFbind8"):
        super().__init__()
        print(f'Loading Oracle Data ...')
        with open(oracle_path, 'rb') as f:
            oracle_data = pickle.load(f)
        if name=='sehstr':
            with open('data_bio/sehstr/sehstr_gbtr_allpreds.pkl', 'rb') as f:
                oracle_data['y'] = pickle.load(f)
        if name=='TFbind10':
            from scipy.special import expit
            oracle_data['y']=  expit(oracle_data['y']* 3 )                          #for tfbind10
        oracle_data['y'] = np.maximum(oracle_data['y'], reward_min)# for qm9str sshe
        oracle_data['y'] =oracle_data['y']**reward_exp  # milden sharpness
        oracle_data['y']=oracle_data['y']* reward_max/oracle_data['y'].max() # scale up
        #oracle_data['y']=torch.maximum(oracle_data['y'],torch.tensor(reward_min)) # scale down  for tf8
        self.O_y = torch.tensor(oracle_data['y'], dtype=torch.float).squeeze()
        self.O_x =torch.tensor(oracle_data['x'],dtype=torch.long)
        self.nbase=nbase
        self.ndim=ndim

        if mode_path is not None:
            with open(mode_path, 'rb') as f:
                modes  = pickle.load(f)
            self.modes = torch.tensor(modes).long()
        else:
            num_modes    = int(len(self.O_y) * 0.001) if name == "sehstr" else int(len(self.O_y) * 0.005) # .005 for qm9str
            sorted_index = torch.sort(self.O_y)[1]
            self.modes   = self.O_x[sorted_index[-num_modes:]]
            self.thresholds = self.O_y[sorted_index[-num_modes:]].min()

    def is_mode(self, states: StatesTensor)-> BatchTensor:
        modes    = nbase2dec(self.nbase, self.modes, self.ndim)
        states   = nbase2dec(self.nbase,states.long(), self.ndim)
        matched  = torch.isin(states,modes)
        return  matched

    def is_mode_r(self, rewards: StatesTensor)-> BatchTensor:
        return  rewards >=self.thresholds

    def __call__(self, states: StatesTensor)-> BatchTensor:
        self.O_y= self.O_y.to(states.device)
        states  = nbase2dec(self.nbase,states.long(),self.ndim)
        reward  = self.O_y[states]
        return reward

class BioSeqEnv(Env):
    def __init__(
        self,
        ndim: int,
        oracle_path,
        mode_path=None,
        alpha:int = 3.0,
        R_max:float  =10.0,
        R_min:float  =1e-3,
        device_str: Literal["cpu", "cuda"] = "cpu",
        preprocessor_name: Literal["KHot"] = "KHot",
        nbase=4,
        name="TFbind8"
    ):
        """Discrete Sequence Graded environment with nbase*ndim actions.

        Args:
            nbase(int, optional): number   N of  integer set  {0,1,.....N}
            ndim (int, optional): dimension D of the sampling space {0,1,...,N}^D.
            alpha (float, optional): scaling factor for oracle. Defaults to 1.0.
            device_str (str, optional): "cpu" or "cuda". Defaults to "cpu".
        """
        self.ndim = ndim
        self.nbase = nbase
        s0 = torch.full((ndim,), -1, dtype=torch.long, device=torch.device(device_str))
        sf = torch.full((ndim,), nbase, dtype=torch.long, device=torch.device(device_str))
        self.oracle  = Oracle(nbase,ndim,
                              oracle_path,mode_path,
                              reward_exp=alpha,reward_max=R_max,reward_min=R_min,name=name)
        action_space = Discrete(nbase * ndim + 1)
        # the last action is the exit action that is only available for complete states
        # Action i in [0, ndim - 1] corresponds to replacing s[i] with 0
        # Action i in [ndim, 2*ndim - 1] corresponds to replacing s[i] with 1
        # Action i in [2*ndim, 3*ndim - 1] corresponds to replacing s[i] with 2
        # .......
        # Action i in [(nbase-1)*ndim, nbase * ndim - 1] corresponds to replacing s[i] with nbase-1

        if preprocessor_name == "KHot":
            preprocessor = KHotPreprocessor(height=nbase, ndim=ndim,fill_value=-1)
        else:
            raise ValueError(f"Unknown preprocessor {preprocessor_name}")

        super().__init__(action_space=action_space,
                         s0=s0, sf=sf,
                         device_str=device_str,
                         preprocessor=preprocessor)

    def make_States_class(self) -> type[States]:
        env = self

        class BioSeqStates(States):
            state_shape: ClassVar[tuple[int, ...]] = (env.ndim,)
            s0 = env.s0
            sf = env.sf

            @classmethod
            def make_random_states_tensor(
                cls, batch_shape: Tuple[int, ...]
            ) -> StatesTensor:
                states_tensor=torch.randint(-1, env.nbase, batch_shape + (env.ndim,), dtype=torch.long, device=env.device)
                return  states_tensor

            def make_masks(self) -> Tuple[ForwardMasksTensor, BackwardMasksTensor]:
                forward_masks = torch.zeros(self.batch_shape + (env.n_actions,),device=env.device,dtype=torch.bool)
                rep_dims=len(self.batch_shape)*(1,)+(env.nbase,)
                forward_masks[..., :-1] = (self.states_tensor == -1).repeat(rep_dims)
                forward_masks[..., -1] =  (self.states_tensor != -1).all(dim=-1)
                #######################
                backward_masks = self.states_tensor.repeat(rep_dims) == \
                                 torch.arange(0, env.nbase, 1,device=env.device).repeat_interleave(env.ndim)
                return forward_masks, backward_masks

            def update_masks(self,action=None,index=None) -> None:
                rep_dims=len(self.batch_shape)*(1,)+(env.nbase,)
                self.forward_masks[...,:-1] = (self.states_tensor == -1).repeat(rep_dims)     # logit are empty we can filled it  by [0,nbase-1].
                self.forward_masks[..., -1] = (self.states_tensor != -1).all(dim=-1)          #when all logits are filled, we can terminating the generating process by s_f
                #######
                self.backward_masks=self.states_tensor.repeat(rep_dims) == \
                                    torch.arange(0, env.nbase, 1,device=env.device).repeat_interleave(env.ndim)  # logit are filled by i=0,..,nbase-1,
                                                                                               # we can take backward actions to remove i and denote the empty logit -1
        return BioSeqStates

    def maskless_step(self, states: StatesTensor, actions: BatchTensor) -> StatesTensor:
        targets= torch.div(actions, self.ndim, rounding_mode='floor') #  fill source slot: 1....,ndim by target digit: 0,....nbase-1
        sources=torch.fmod(actions, self.ndim)                        #  [digit 0: slot 1,....,slot ndim  ;.....;digit nbase-1: slot 1,..., slot ndim]
        return  states.scatter_(-1, sources.unsqueeze(-1), targets.unsqueeze(-1))

    def maskless_backward_step(self, states: StatesTensor, actions: BatchTensor) -> StatesTensor:
        sources= torch.fmod(actions, self.ndim)                         #  sources: state element index
        return states.scatter_(-1, sources.unsqueeze(-1), -1)           #  target: -1

    def get_states_indices(self, states: States) -> BatchTensor:
        """The chosen encoding is the following: -1 -> 0, 0 -> 1, 1 -> 2,.... then we convert to base 5"""
        return nbase2dec(self.nbase+1, states.states_tensor+1,self.ndim).long().cpu().tolist()

    def get_terminating_states_indices(self, states: States) -> BatchTensor:
        return nbase2dec(self.nbase, states.states_tensor,self.ndim).long().cpu().tolist()

    @property
    def n_states(self) -> int:
        return (self.nbase+1)**self.ndim

    @property
    def n_terminating_states(self) -> int:
        return self.nbase**self.ndim

    @property
    def all_states(self) -> States:
        # This is brute force !
        if self.n_states>3e7:
            raise ValueError('too large (>{}) to enumerate all possible states'.format(3e7))
        digits = torch.arange(self.nbase+1, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.ndim)# add -1 for s_f state
        all_states = all_states - 1
        return self.States(all_states)


    @property
    def ordered_states(self) -> States:
        # This is brute force !
        ordered_states = self.all_states
        index=[]
        for i in reversed(range(self.ndim+1)):
            index.append(torch.where(torch.sum(ordered_states.states_tensor==-1,-1) ==i)[0])
        return ordered_states[torch.cat(index)]

    @property
    def ordered_states_list(self) -> List[States]:
        # This is brute force !
        ordered_states = self.all_states
        index=[]
        for i in reversed(range(self.ndim+1)):
            index.append(torch.where(torch.sum(ordered_states.states_tensor==-1,-1) ==i)[0])
        return [ordered_states[ind] for ind in index]

    @property
    def terminating_states(self) -> States:
        if self.n_states>1e7:
            raise ValueError('too large (>{}) to enumerate all possible states'.format(1e7))
        digits = torch.arange(self.nbase, device=self.device)
        all_states = torch.cartesian_prod(*[digits] * self.ndim)
        return self.States(all_states)

    def log_reward(self, final_states: States) -> BatchTensor:
        raw_states = final_states.states_tensor
        return self.oracle(raw_states).log()

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        log_reward=self.log_reward(self.terminating_states)
        true_dist = log_reward-torch.logsumexp(log_reward,-1)
        return true_dist.exp().cpu()

    @property
    def mean_reward(self) -> torch.float:
        return (self.oracle.O_y ** 2).sum() / self.oracle.O_y.sum()

    @property
    def log_partition(self) -> torch.float:
        return (self.oracle.O_y.sum()).log()
