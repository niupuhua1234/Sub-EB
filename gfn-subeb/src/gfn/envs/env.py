from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
from gymnasium.spaces import Discrete, Space
from torchtyping import TensorType

from src.gfn.containers.states import States
from src.gfn.envs.preprocessors import IdentityPreprocessor, Preprocessor

# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]  #
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]# -1 as without exit action
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]
PmfTensor = TensorType["n_states", torch.float]

NonValidActionsError = type("NonValidActionsError", (ValueError,), {})


class Env(ABC):
    """
    Base class for environments, showing which methods should be implemented.
    A common assumption for all environments is that all actions are discrete,
    represented by a number in {0, ..., n_actions - 1}, the last one being the
    exit action.
    """


    def __init__(
        self,
        action_space: Space,
        s0: OneStateTensor,
        sf: Optional[OneStateTensor] = None,
        device_str: Optional[str] = None,
        preprocessor: Optional[Preprocessor] = None,
        bction_space: Optional[Space]=None,
    ):
        self.s0 = s0
        self.device = torch.device(device_str) if device_str is not None else s0.device
        self.sf = sf if sf is not None else torch.full(s0.shape, -float("inf"),device=self.device)
        self.action_space = action_space
        self.bction_space = Discrete(action_space.n-1) if bction_space is None else bction_space
        self.States = self.make_States_class()
        self.preprocessor = preprocessor if preprocessor is not None else IdentityPreprocessor(output_shape=(s0.shape))
        self.logZ_est=torch.tensor(0.,dtype=torch.float)

    def to_device(self,device:str):
        self.s0=self.s0.to(device)
        self.sf=self.sf.to(device)
        self.States.s0=self.States.s0.to(device)
        self.States.sf=self.States.sf.to(device)
        self.device=torch.device(device)

    @abstractmethod
    def make_States_class(self) -> type[States]:
        "Returns a class that inherits from States and implements the environment-specific methods."
        pass

    def is_sink_actions(self, actions: TensorLong) -> TensorBool:
        "Returns True if the action is an sink action."
        return actions==-1

    def is_exit_actions(self, actions: TensorLong) -> TensorBool:
        "Returns True if the action is an exit action."
        return actions == self.n_actions - 1

    @abstractmethod
    def maskless_step(self, states: StatesTensor, actions: TensorLong) -> None:
        """Same as the step function, but without worrying whether or not the actions are valid, or masking."""
        pass

    @abstractmethod
    def maskless_backward_step(self, states: StatesTensor, actions: TensorLong) -> None:
        """Same as the backward_step function, but without worrying whether or not the actions are valid, or masking."""
        pass

    def bction2action(self,states: States, bctions: TensorLong)->TensorLong:
        "When the forward actions  is difffent from the backward, this infer forward actions from backward actions"
        return bctions

    def action2bction(self,states: States, actions: TensorLong)->TensorLong:
        "When the forward actions  is difffent from the backward, this infer forward actions from backward actions"
        return actions

    def log_reward(self, final_states: States) -> TensorFloat:
        """Either this or reward needs to be implemented."""
        raise NotImplementedError("log_reward function not implemented")

    def get_states_indices(self, states: States):# -> TensorLong:
        return NotImplementedError("The environment does not support enumeration of states")

    def get_terminating_states_indices(self, states: States):# -> TensorLong:
        return NotImplementedError("The environment does not support enumeration of states")

    @property
    def n_actions(self) -> int:
        if isinstance(self.action_space, Discrete):
            return self.action_space.n
        else:
            raise NotImplementedError("Only discrete action spaces are supported")

    @property
    def n_bctions(self) -> int:
        if isinstance(self.bction_space, Discrete):
            return self.bction_space.n
        else:
            raise NotImplementedError("Only discrete action spaces are supported")

    @property
    def n_states(self):# -> int:
        return NotImplementedError("The environment does not support enumeration of states")

    @property
    def n_terminating_states(self):# -> int:
        return NotImplementedError(
            "The environment does not support enumeration of states"
        )


    @property
    def true_dist_pmf(self):# -> PmfTensor:
        "Returns a one-dimensional tensor representing the true distribution."
        return NotImplementedError( "The environment does not support enumeration of states")

    @property
    def log_partition(self):# -> float:
        "Returns the logarithm of the partition function."
        return NotImplementedError( "The environment does not support enumeration of states")

    @property
    def all_states(self):# -> States:
        """Returns a batch of all states for environments with enumerable states.
        The batch_shape should be (n_states,).
        This should satisfy:
        self.get_states_indices(self.all_states) == torch.arange(self.n_states)
        """
        return NotImplementedError("The environment does not support enumeration of states")

    @property
    def terminating_states(self):# -> States:
        """Returns a batch of all terminating states for environments with enumerable states.

        The batch_shape should be (n_terminating_states,).This should satisfy:

        ``self.get_terminating_states_indices(self.terminating_states) == torch.arange(self.n_terminating_states)``
        """
        return NotImplementedError("The environment does not support enumeration of states")

    def reset(
        self, batch_shape: Union[int, Tuple[int]], random: bool = False
    ) -> States:
        "Instantiates a batch of initial states."
        if isinstance(batch_shape, int):batch_shape = (batch_shape,)
        return self.States.from_batch_shape(batch_shape=batch_shape, random=random)

    def step(
            self,
            states: States,
            actions: TensorLong
    ) -> States:
        """
        Function that takes a batch of states and actions and returns a batch of next states
        and a boolean tensor indicating sink states in the new batch.(Noet: for one time step)
        """

        #terminating states in the past
        new_states = deepcopy(states)
        valid_index: TensorBool = ~states.is_sink_state

        # check whthere action are taken over masked states
        if new_states.forward_masks is not None:
            valid_states_masks = new_states.forward_masks[valid_index]
            valid_actions_bool = all( torch.gather(valid_states_masks, 1, actions[valid_index].unsqueeze(1)))
            if not valid_actions_bool:
                raise NonValidActionsError("Some Actions are not valid, check action sampler")

        # new_sink_state induced by the current actions
        new_sink_index = self.is_exit_actions(actions)
        new_states.states_tensor[new_sink_index] = self.sf
        new_valid_index = valid_index & ~new_sink_index # update invalid states and sink states

        new_states.states_tensor[new_valid_index]=\
            self.maskless_step(new_states.states_tensor[new_valid_index], actions[new_valid_index]) # update states
        new_states.update_masks(actions,new_valid_index)
        return new_states

    def all_step(
            self,
            states: States,Backward=False) -> States:
        #terminating states in the past
        action_space=self.bction_space.n if Backward else self.action_space.n
        all_states_tensor=states.states_tensor.unsqueeze(-2).repeat(*len(states.batch_shape)*(1,),  action_space,1)
        all_forward_mask=states.forward_masks.unsqueeze(-2).repeat(*len(states.batch_shape)*(1,),  action_space,1)
        all_backward_mask=states.backward_masks.unsqueeze(-2).repeat(*len(states.batch_shape)*(1,),  action_space,1)

        all_states=self.States(all_states_tensor,all_forward_mask,all_backward_mask)
        actions = torch.arange(action_space).repeat(*states.batch_shape, 1)

        if not Backward:
            # assign all masked actions to be terminating action, this render sf in new states
            all_states.states_tensor[~states.forward_masks]=self.sf
            new_states=self.step(all_states,actions)
        else:
            all_states.states_tensor[~states.backward_masks] = self.s0
            new_states=self.backward_step(all_states,actions)
        return new_states

    def backward_step(self, states: States, actions: TensorLong) -> States:
        """Function that takes a batch of states and actions and returns a batch of next
        states and a boolean tensor indicating initial states in the new batch."""
        new_states = deepcopy(states)
        valid_index: TensorBool = ~new_states.is_initial_state &~new_states.is_sink_state
        if new_states.backward_masks is not None:
            valid_states_masks =  new_states.backward_masks[valid_index]
            valid_actions_bool = all(torch.gather(valid_states_masks, 1,actions[valid_index].unsqueeze(1)))
            if not valid_actions_bool:
                raise NonValidActionsError("Actions are not valid")

        new_states.states_tensor[valid_index]=\
            self.maskless_backward_step(new_states.states_tensor[valid_index],actions[valid_index])
        new_states.backward_update_masks(actions,valid_index)
        return new_states
