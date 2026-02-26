from __future__ import annotations

from abc import ABC
from math import prod
from typing import ClassVar, Sequence, cast,Optional

import torch
from torchtyping import TensorType

from src.gfn.containers.base import Container

# Typing
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]
DonesTensor = TensorType["batch_shape", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
TensorLong = TensorType["batch_shape", torch.long]
TensorBool = TensorType["batch_shape", torch.bool]

class States(Container, ABC):
    """
    Base class for states, seen as nodes of the DAG.
    For each environment, a States subclass is needed. A `States` object
    is a collection of multiple states (nodes of the DAG). A tensor representation
    of the states is required for batching. If a state is represented with a tensor
    of shape (*state_shape), a batch of states is represented with a States object,
    with the attribute `states_tensor` of shape (*batch_shape, *state_shape). Other
    representations are possible (e.g. state as string, as numpy array, as graph, etc...),
    but these representations should not be batched.


    If the environment's action space is discrete, then each States object is also endowed
    with a `forward_masks` and `backward_masks` boolean attributes representing which actions
    are allowed at each state.

    A `batch_shape` attribute is also required, to keep track of the batch dimension.
    A trajectory can be represented by a States object with batch_shape = ``(n_states,)``.
    Multiple trajectories can be represented by a States object with batch_shape = ``(n_states, n_trajectories)``.

    Because multiple trajectories can have different lengths, batching requires appending a dummy tensor
    to trajectories that are shorter than the longest trajectory. The dummy state is the ``s_f``
    attribute of the environment (e.g. [-1, ..., -1], or [-inf, ..., -inf], etc...). Which is never processed,
    and is used to pad the batch of states only.

    Args:
        states_tensor (StatesTensor): A tensor representatio of the states is required for batching.
        If a state is represented with a tensor of shape (*state_shape), a batch of states is represented with a States object,
        with the attribute `states_tensor` of shape (*batch_shape, *state_shape)
        forward_masks (ForwardMasksTensor): representing which actions are allowed at each state.( Used for action_Sampler).
                                            s_f doesn't need to be masked as neuralnetwork input exclude s_f during traj sampling and loss computation.
                                            s_0 need to be masked  as it is valid input within forward traj.
        backward_masks (BackwardMasksTensor):representing which actions are allowed at each state.
    """
    #公共属性 under this class
    state_shape: ClassVar[tuple[int, ...]]  # Shape of one state  #class variabe
    s0: ClassVar[OneStateTensor]            # Source state of the DAG
    sf: ClassVar[OneStateTensor]            # Dummy state, used to pad a batch of states

    def __init__(
        self,
        states_tensor: StatesTensor,
        forward_masks: ForwardMasksTensor | None = None,
        backward_masks: BackwardMasksTensor | None = None,
    ):
        self.states_tensor = states_tensor
        self.batch_shape = tuple(self.states_tensor.shape)[: -len(self.state_shape)]
        if forward_masks is None and backward_masks is None:
            self.forward_masks, self.backward_masks = self.make_masks()
        else:
            self.forward_masks,self.backward_masks = forward_masks,backward_masks

    def to_device(self,device:str):
        for key, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                self.__setattr__(key, self.__getattribute__(key).to(device))

    #可以在不实例化的情形下访问该方法， 相当于一种实例化操作的装饰
    @classmethod
    def from_batch_shape(cls, batch_shape: tuple[int], random: bool = False) -> States:
        """Create a States object with the given batch shape, all initialized to s_0.
        If random is True, the states are initialized randomly. This requires that
        the environment implements the `make_random_states_tensor` class method.
        """
        if random:
            states_tensor = cls.make_random_states_tensor(batch_shape)
        else:
            states_tensor = cls.make_initial_states_tensor(batch_shape)
        return cls(states_tensor)

    @classmethod
    def make_initial_states_tensor(cls, batch_shape: tuple[int]) -> StatesTensor:
        state_ndim = len(cls.state_shape)
        assert cls.s0 is not None and state_ndim is not None
        return cls.s0.repeat(*batch_shape, *((1,) * state_ndim))

    @classmethod
    def make_random_states_tensor(cls, batch_shape: tuple[int]) -> StatesTensor:
        raise NotImplementedError(
            "The environment does not support initialization of random states."
        )

    def make_masks(self): #-> tuple[ForwardMasksTensor, BackwardMasksTensor]:
        """Create the forward and backward masks for the states.
        This method is called only if the masks are not provided at initialization.
        """
        return NotImplementedError(
            "make_masks method not implemented. Your environment must implement it if discrete"
        )

    def update_masks(self,actions:Optional[TensorLong],index:Optional[TensorBool]):# -> None:
        """Update the masks, if necessary.
        This method should be called after each action is taken.
        """
        return NotImplementedError( "update_masks method not implemented. Your environment must implement it if discrete")

    def backward_update_masks(self, actions: Optional[TensorLong], index: Optional[TensorBool]):  # -> None:
        """Update the masks, if necessary.
        This method should be called after each action is taken.
        """
        self.update_masks(actions,index)

    def __len__(self):
        return prod(self.batch_shape)

    def __repr__(self):
        return f"{self.__class__.__name__} object of batch shape {self.batch_shape} and state shape {self.state_shape}"

    @property
    def device(self) -> torch.device:
        return self.states_tensor.device

    def __setitem__(self, index: int | Sequence[int] | Sequence[bool], value: States) -> None:
        """
        Replace part of this States object at `index` with the contents of another `States` object.
        This supports assignment like `states[i] = other_states`.
        """
        if not isinstance(value, States):raise TypeError("Assigned value must be a `States` instance.")
        # Assign main tensor
        self.states_tensor[index] = value.states_tensor
        for key, val in self.__dict__.items() :
            if key != "states_tensor" and isinstance(val,torch.Tensor):
                # Assign masks if available
                if val is not None and value.__getattribute__(key) is not None:
                    val[index] = value.__getattribute__(key)
                    self.__setattr__(key,val)

    def __getitem__(self, index: int | Sequence[int] | Sequence[bool]) -> States:
        """Access particular states of the batch."""
        kwargs= {}
        states = self.states_tensor[index]
        for key, val in self.__dict__.items() :
            if key != "states_tensor" and isinstance(val,torch.Tensor):
                kwargs[key] = val[index]

        return self.__class__( states, **kwargs)

    def flatten(self) -> States:
        """Flatten the batch dimension of the states.
        This is useful for example when extracting individual states from trajectories.
        """
        kwargs = {}
        states = self.states_tensor.view(-1, *self.state_shape)
        for key, val in self.__dict__.items() :
            if key != "states_tensor" and isinstance(val,torch.Tensor):
                new_val = val.view(-1, val.shape[-1])
                kwargs[key] = new_val
        return self.__class__(states, **kwargs)


    def extend(self, other: States) -> None:
        """Collates to another States object of the same batch shape, which should be 1 or 2.
        If 1, this corresponds to connect a trajectory to the end of the  trajectory.
        If 2, this corresponds to merge two batch of trajectory in batch-wise dim

        Args:
            other (States): Batch of states to collate to.

        Raises:
            ValueError: if self.batch_shape != other.batch_shape or if self.batch_shape != (1,) or (2,)
        """
        other_batch_shape = other.batch_shape
        if len(other_batch_shape) == len(self.batch_shape) == 1:
            # This corresponds to adding a state to a trajectory
            self.batch_shape = (self.batch_shape[0] + other_batch_shape[0],)
            self.states_tensor = torch.cat(  (self.states_tensor, other.states_tensor), dim=0)

        elif len(other_batch_shape) == len(self.batch_shape) == 2:
            # This corresponds to adding a trajectory to a batch of trajectories
            self.extend_with_sf( max(self.batch_shape[0], other_batch_shape[0]))
            other.extend_with_sf(max(self.batch_shape[0], other_batch_shape[0]))
            self.batch_shape = (self.batch_shape[0],self.batch_shape[1] + other_batch_shape[1],)
            self.states_tensor = torch.cat( (self.states_tensor, other.states_tensor), dim=1)
        else:
            raise ValueError( f"extend is not implemented for batch shapes "
                              f"{self.batch_shape} and {other_batch_shape}")

        for key, val in self.__dict__.items() :
            if key != "states_tensor" and isinstance(val,torch.Tensor):
                val = torch.cat((val,other.__getattribute__(key)), dim=len(self.batch_shape) - 1)
                self.__setattr__(key,val)

    def extend_with_sf(self, required_first_dim: int) -> None:
        """Takes a two-dimensional batch of states (i.e. of batch_shape (a, b)),
        and extends it to a States object of batch_shape (required_first_dim, b),
        by adding the required number of `s_f` tensors. This is useful to extend trajectories
        of different lengths."""
        if len(self.batch_shape) == 2:
            if self.batch_shape[0] >= required_first_dim:
                return
            padding_shape=required_first_dim - self.batch_shape[0]
            self.states_tensor = torch.cat((self.states_tensor,self.sf.repeat(padding_shape,self.batch_shape[1], 1)),dim=0)
            for key, val in self.__dict__.items():
                if key != "states_tensor" and isinstance(val, torch.Tensor):
                    val = torch.cat((val,torch.ones(padding_shape,*val.shape[1:],dtype=torch.bool,device=self.device)),dim=0)
                    self.__setattr__(key, val)

            self.batch_shape = (required_first_dim, self.batch_shape[1])
        else:
            raise ValueError( f"extend_with_sf is not implemented for batch shapes {self.batch_shape}")

    def compare(self, other: StatesTensor) -> DonesTensor:
        """Given a tensor of states, returns a tensor of booleans indicating whether the states
        are equal to the states in self.

        Args:
            other (StatesTensor): Tensor of states to compare to.

        Returns:
            DonesTensor: Tensor of booleans indicating whether the states are equal to the states in self.
        """
        out = self.states_tensor == other.to(self.device)
        state_ndim = len(self.__class__.state_shape)
        for _ in range(state_ndim):
            out = out.all(dim=-1) # recursive checking from last state_axis to the first state_axis
        return out

    @property
    def is_initial_state(self) -> DonesTensor:
        r"""Return a boolean tensor of shape=(*batch_shape,),
        where True means that the state is $s_0$ of the DAG.
        """
        source_states_tensor = self.__class__.s0.repeat(
            *self.batch_shape, *((1,) * len(self.__class__.state_shape))
        )
        return cast(DonesTensor,self.compare(source_states_tensor))

    @property
    def is_sink_state(self) -> DonesTensor:
        r"""Return a boolean tensor of shape=(*batch_shape,),
        where True means that the state is $s_f$ of the DAG.
        """
        sink_states = self.__class__.sf.repeat(
            *self.batch_shape, *((1,) * len(self.__class__.state_shape)))  # *(time,btach)+ (1,)* 1
        return cast(DonesTensor,self.compare(sink_states))

