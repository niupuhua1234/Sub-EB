from abc import ABC, abstractmethod
from typing import ClassVar, Literal, Tuple, cast,List
import torch
from gymnasium.spaces import Discrete
from torchtyping import TensorType
from src.gfn.containers.states import States
from src.gfn.envs.env import Env
from src.gfn.envs.preprocessors import KHotPreprocessor
# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]
BatchTensor = TensorType["batch_shape"]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
from src.gfn.envs.bioseq import Oracle

def nbase2dec(n,b, length):
    #n:n_base b:bits
    canonical_base = n ** torch.arange(length).to(b.device, b.dtype)
    return torch.sum(canonical_base * b, -1)

class BioSeqPendEnv(Env):
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
        """Discrete Sequence Graded environment with 2*nbase actions.

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
        action_space = Discrete(2*nbase + 1)
        bction_space = Discrete(2)
        # the last action is the exit action that is only available for complete states
        # Action  i ∈ [0, nbase - 1]  corresponds to append s with i,
        # Action  i ∈ [nbase, 2base-1] corresponds to prepend s with i
        # Action 2nbase               corresponds to action (s->sf)
        # Bction =0                   corresponds to de-append digits, i.e. remove the last digit
        # Bction =1                  corresponds to de-prepend digits, i.e. remove the first digit
        if preprocessor_name == "KHot":
            preprocessor = KHotPreprocessor(height=nbase, ndim=ndim,fill_value=-1)
        else:
            raise ValueError(f"Unknown preprocessor {preprocessor_name}")

        super().__init__(action_space=action_space,
                         bction_space=bction_space,
                         s0=s0, sf=sf,
                         device_str=device_str,
                         preprocessor=preprocessor,)

    def make_States_class(self) -> type[States]:
        env = self
        class BioSeqStates(States):
            state_shape: ClassVar[tuple[int, ...]] = (env.ndim,)
            s0 = env.s0
            sf = env.sf
            action_base=env.nbase * torch.arange(0,env.ndim+1,1)
            @classmethod
            def make_random_states_tensor(
                cls, batch_shape: Tuple[int, ...]
            ) -> StatesTensor:
                states_tensor=torch.randint(-1, 2, batch_shape + (env.ndim,), dtype=torch.long, device=env.device)
                return  states_tensor

            def make_masks(self) -> Tuple[ForwardMasksTensor, BackwardMasksTensor]:
                "Mask illegal (forward and backward) actions."
                rep_dims = len(self.batch_shape) * (1,) + (2*env.nbase,)
                forward_masks = torch.ones((*self.batch_shape, env.n_actions),dtype=torch.bool,device=env.device)
                forward_masks[..., :-1] = (self.states_tensor==-1).any(-1,keepdim=True).repeat(rep_dims) # sequences are not complete, we append/prepend it by i ∈[0,nabse-1]
                forward_masks[..., -1]  = (self.states_tensor !=-1).all(-1)  # sequences are complete, the process is terminated by transiting to s_f
                rep_dims = len(self.batch_shape) * (1,) + (2,)
                backward_masks          = (~self.is_initial_state).unsqueeze(-1).repeat(rep_dims)  # sequences are not empty, we can remove the last/first digits.
                return forward_masks, backward_masks

            def update_masks(self,action=None,index=None) -> None:
                "Update the masks based on the current states."
                rep_dims = len(self.batch_shape) * (1,) + (2 * env.nbase,)
                self.forward_masks[...,:-1] = (self.states_tensor==-1).any(-1,keepdim=True).repeat(rep_dims)
                self.forward_masks[..., -1] = (self.states_tensor !=-1).all(-1)
                rep_dims = len(self.batch_shape) * (1,) + (2,)
                self.backward_masks          = (~self.is_initial_state).unsqueeze(-1).repeat(rep_dims)
        return BioSeqStates

    def bction2action(self,states: States, bctions: TensorLong) ->TensorLong:
        actions=torch.full_like(bctions,fill_value=-1)
        last_index         = self.backward_index(states.states_tensor[bctions==0])
        actions[bctions==0]=states.states_tensor[bctions==0,last_index]  #bction=0:  action=last digit
        actions[bctions==1]=states.states_tensor[bctions==1,      [0] ]+self.nbase#baction=1:   action= nbase+ first digit
        return actions
    def action2bction(self,states: States, actions: TensorLong) ->TensorLong:
        bctions=torch.div(actions, self.nbase, rounding_mode='floor')   #bction =action//nbase  =0 or 1
        return bctions

    def forward_index(self,states:StatesTensor ) -> BatchTensor:
        #  use argmin to find the first -1 element
        #  If the seq is complete and there is no -1, then return the first element.
        return (states > -1).int().argmin(-1) # or return (states <= -1).int().argmax(-1)

    def backward_index(self,states:StatesTensor ) -> BatchTensor:
        #  use argmin to find the last non -1 element
        #  If the seq is complete and there is no -1, then return the last element.
        return (states > -1).int().argmin(-1)-1

    def maskless_step(self, states: StatesTensor, actions: BatchTensor) -> StatesTensor:
        new_states = states.clone()
        sources= torch.div(actions, self.nbase, rounding_mode='floor')
        targets= torch.fmod(actions, self.nbase)
        new_states[sources ==0,self.forward_index(states[sources ==0])]     = targets[sources ==0]  #append
        new_states[sources ==1,1:] = states[sources ==1,:-1]
        new_states[sources ==1,0]  = targets[sources ==1]                                          #prepend
        return new_states

    def maskless_backward_step(self, states: StatesTensor, actions: BatchTensor) -> StatesTensor:
        new_states=states.clone()
        new_states[actions ==0,self.backward_index(states[actions ==0])]  = -1   #de-append
        new_states[actions ==1,:-1] = states[actions ==1,1:]
        new_states[actions ==1,-1] = -1   #de-prepend
        return new_states

    def get_states_indices(self, states: States) -> BatchTensor:
        """The chosen encoding is the following: -1 -> 0, 0 -> 1, 1 -> 2,.... then we nbase-1 convert to nbase """
        return nbase2dec(self.nbase, states.states_tensor+1,self.ndim).long().cpu().tolist()

    def get_terminating_states_indices(self, states: States) -> BatchTensor:
        return nbase2dec(self.nbase, states.states_tensor,self.ndim).long().cpu().tolist()

    @property
    def n_states(self) -> int:
        return sum([self.nbase**i for i in range(self.ndim+1)])
    def n_ordered_states(self, order) -> int:
        return sum([self.nbase**i for i in range(order+1)])
    @property
    def n_terminating_states(self) -> int:
        return self.nbase**self.ndim
    def ordered_state_tensor_list(self,min_ndim=0) -> List[StatesTensor]:
        # This is brute force !
        if self.n_states>3e9:
            raise ValueError('too large (>{}) to enumerate all possible states'.format(3e7))
        digits = torch.arange(self.nbase, device=self.device)
        ordered_states=[]
        if min_ndim<=0:
            ordered_states.append(-torch.ones(self.ndim,dtype=torch.long))
        if min_ndim<=1:
            first_states=torch.cartesian_prod(*[digits]).unsqueeze(-1)
            padding_states = - torch.ones(self.nbase, self.ndim - 1, dtype=torch.long)
            ordered_states.append(torch.cat([first_states,  padding_states],dim=-1))
        if self.ndim>1:
            for i in range(max(min_ndim,2),self.ndim+1):
                digit_states = torch.cartesian_prod(*[digits] * i)
                padding_states = - torch.ones(digit_states.shape[0], self.ndim - i, dtype=torch.long)
                add_states = torch.cat([digit_states, padding_states], dim=-1)
                ordered_states.append(add_states)
            return ordered_states
        else:
            return ordered_states
    @property
    def ordered_states_list(self) -> List[States]:
        # This is brute force !
        return [self.States(state_tensors) for state_tensors in self.ordered_state_tensor_list()]
    @property
    def ordered_states(self) -> States:
        # This is brute force !
        return  self.States(torch.vstack(self.ordered_state_tensor_list()))
    @property
    def all_states(self) -> States:
        # This is brute force !
        return self.ordered_states

    @property
    def terminating_states(self) -> States:
        return self.States(torch.vstack(self.ordered_state_tensor_list(self.ndim)))

    def log_reward(self, final_states: States) -> BatchTensor:
        raw_states = final_states.states_tensor
        return self.oracle(raw_states).log()

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        log_reward=self.log_reward(self.terminating_states)
        true_dist = log_reward-torch.logsumexp(log_reward,-1)
        return true_dist.exp().cpu()
    @property
    def mean_reward(self)->torch.float:
        return (2 * self.log_reward(self.terminating_states)-self.log_partition).exp().sum()
    @property
    def log_partition(self) -> torch.float:
        log_rewards = self.log_reward(self.terminating_states)
        return torch.logsumexp(log_rewards, -1)