"""
Copied and Adapted from https://github.com/Tikquuss/GflowNets_Tutorial
"""

from typing import ClassVar, Literal, Tuple, cast

import torch
from einops import rearrange
from gymnasium.spaces import Discrete
from torchtyping import TensorType

from src.gfn.containers.states import States
from src.gfn.envs.env import Env
from src.gfn.envs.preprocessors import KHotPreprocessor

# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]


class HyperGrid(Env):
    def __init__(
        self,
        ndim: int = 2,
        height: int = 4,
        R0: float = 0.1,
        R1: float = 0.5,
        R2: float = 2.0,
        reward_cos: bool = False,
        device_str: Literal["cpu", "cuda"] = "cpu",
        preprocessor_name: Literal["KHot"] = "KHot",
    ):
        """HyperGrid environment from the GFlowNets paper.
        The states are represented as 1-d tensors of length `ndim` with values in
        {0, 1, ..., height - 1}.
        A preprocessor transforms the states to the input of the neural network,
        which can be a one-hot, a K-hot, or an identity encoding.

        Args:
            ndim (int, optional):
                dimension of the grid. Defaults to 2.
            height (int, optional): height of the grid. Defaults to 4.
            R0 (float, optional): reward parameter R0. Defaults to 0.1.
            R1 (float, optional): reward parameter R1. Defaults to 0.5.
            R2 (float, optional): reward parameter R1. Defaults to 2.0.
            reward_cos (bool, optional): Which version of the reward to use. Defaults to False.
            device_str (str, optional): "cpu" or "cuda". Defaults to "cpu".
            preprocessor_name (str, optional): "KHot" or "OneHot" or "Identity". Defaults to "KHot".
        """
        self.ndim = ndim
        self.height = height
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.reward_cos = reward_cos

        s0 = torch.zeros(ndim, dtype=torch.long, device=torch.device(device_str))
        sf = torch.full( (ndim,), fill_value=-1, dtype=torch.long, device=torch.device(device_str))

        action_space = Discrete(ndim + 1)

        if preprocessor_name == "KHot":
            preprocessor = KHotPreprocessor(height=height, ndim=ndim)
        else:
            raise ValueError(f"Unknown preprocessor {preprocessor_name}")

        super().__init__(
            action_space=action_space,
            s0=s0,
            sf=sf,
            device_str=device_str,
            preprocessor=preprocessor,
        )
    def make_States_class(self) -> type[States]:
        "Creates a States class for this environment"
        env = self
        class HyperGridStates(States):
            state_shape: ClassVar[tuple[int, ...]] = (env.ndim,)
            s0:ClassVar[OneStateTensor]            = env.s0
            sf:ClassVar[OneStateTensor]            = env.sf
            @classmethod
            def make_random_states_tensor(cls, batch_shape: Tuple[int, ...]) -> StatesTensor:
                "Creates a batch of random states."
                states_tensor = torch.randint(0, env.height, batch_shape + env.s0.shape, device=env.device) # range=(0-height), size=(batch_shape,)+(state_shape,)
                return states_tensor

            def make_masks(self) -> Tuple[ForwardMasksTensor, BackwardMasksTensor]:
                "Mask illegal (forward and backward) actions."
                forward_masks = torch.ones((*self.batch_shape, env.n_actions),dtype=torch.bool,device=env.device)
                forward_masks[..., :-1] = self.states_tensor != env.height - 1 # When reching the grid bound, masking all actions but terminating action
                backward_masks = self.states_tensor != 0                       # Maksing the zero states
                return forward_masks, backward_masks

            def update_masks(self,action=None,index=None) -> None:
                "Update the masks based on the current states."
                # The following two lines are for typing only.
                self.forward_masks[..., :-1] = self.states_tensor != env.height - 1
                self.backward_masks = self.states_tensor != 0
        return HyperGridStates

    def maskless_step(self, states: StatesTensor, actions: TensorLong)->StatesTensor:
        return states.scatter_(-1, actions.unsqueeze(-1), 1, reduce="add")

    def maskless_backward_step(self, states: StatesTensor, actions: TensorLong)->StatesTensor:
        return states.scatter_(-1, actions.unsqueeze(-1), -1, reduce="add")

    def true_reward(self, final_states: States) -> TensorFloat:
        final_states_raw = final_states.states_tensor
        R0, R1, R2 = (self.R0, self.R1, self.R2)
        ax = abs(final_states_raw / (self.height - 1) - 0.5)
        if not self.reward_cos:
            reward = (R0 + (0.25 < ax).prod(-1) * R1 + ((0.3 < ax) * (ax < 0.4)).prod(-1) * R2)
        else:
            pdf_input = ax * 5
            pdf = 1.0 / (2 * torch.pi) ** 0.5 * torch.exp(-(pdf_input**2) / 2)
            reward = R0 + ((torch.cos(ax * 50) + 1) * pdf).prod(-1) * R1
        return reward

    def log_reward(self, final_states: States) -> TensorFloat:
        log_reward= torch.log(self.true_reward(final_states))
        return log_reward

    def build_grid(self) -> States:
        "Utility function to build the complete grid"
        H = self.height
        ndim = self.ndim
        grid_shape = (H,) * ndim + (ndim,)  # (H, ..., H, ndim)
        grid = torch.zeros(grid_shape, device=self.device)
        for i in range(ndim):
            grid_i = torch.linspace(start=0, end=H - 1, steps=H)
            for _ in range(i):
                grid_i = grid_i.unsqueeze(1)
            grid[..., i] = grid_i

        rearrange_string = " ".join([f"n{i}" for i in range(1, ndim + 1)])
        rearrange_string += " ndim -> "
        rearrange_string += " ".join([f"n{i}" for i in range(ndim, 0, -1)])
        rearrange_string += " ndim"
        grid = rearrange(grid, rearrange_string).long()
        return self.States(grid)

    def get_states_indices(self, states: States):
        states_raw = states.states_tensor
        canonical_base = self.height ** torch.arange(self.ndim - 1, -1, -1, device=states_raw.device)
        return (canonical_base * states_raw).sum(-1).long().cpu().tolist()

    def get_terminating_states_indices(self, states: States):
        return self.get_states_indices(states)

    @property
    def n_states(self) -> int:
        return self.height**self.ndim

    @property
    def n_terminating_states(self) -> int:
        return self.n_states


    @property
    def all_states(self) -> States:
        grid = self.build_grid()
        flat_grid = rearrange(grid.states_tensor, "... ndim -> (...) ndim")  # (height,height,ndim)-> (height*hieght,ndim)
        return self.States(flat_grid)

    @property
    def terminating_states(self) -> States:
        return self.all_states

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        #assert self.get_states_indices(all_states)== torch.arange(self.n_states, device=self.device).tolist()
        reward = self.true_reward(self.terminating_states)
        true_dist = reward/reward.sum()
        return true_dist.cpu()
    @property
    def log_partition(self) -> float:
        all_states = self.all_states
        rewards = self.true_reward(all_states)
        return rewards.sum().log()

    @property
    def mean_reward(self)->torch.float:
        return (self.true_dist_pmf * self.true_reward(self.terminating_states)).sum()

