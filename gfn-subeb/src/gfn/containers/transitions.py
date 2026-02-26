from __future__ import annotations

from typing import TYPE_CHECKING, Sequence,Optional

import torch
from torchtyping import TensorType

if TYPE_CHECKING:
    from src.gfn.envs import Env
    from src.gfn.containers.states import States

from src.gfn.containers.base import Container

# Typing  -- n_transitions is either int or Tuple[int]
LongTensor = TensorType["n_transitions", torch.long]
BoolTensor = TensorType["n_transitions", torch.bool]
FloatTensor = TensorType["n_transitions", torch.float]
PairFloatTensor = TensorType["n_transitions", 2, torch.float]


class Transitions(Container):
    def __init__(
        self,
        env: Env,
        states:  Optional[States] = None,
        actions: Optional[LongTensor] = None,
        is_done: Optional[BoolTensor] = None,
        next_states: Optional[States] = None,
        is_backward: bool = False,
        log_probs: Optional[FloatTensor] = None,
        n_trajectories=1,
    ):
        """
        Container for transitions.
        When states and next_states are not None, the Transitions is an empty container that can be populated on the go.

        Args:
            env (Env): Environment
            states (States, optional): States object with `uni-dimensional` batch_shape, representing the parents of the transitions. Defaults to None.
            actions (LongTensor, optional): Actions chosen at the parents of each transition.   Defaults to None.
            is_done (BoolTensor, optional): Whether the action is the exit action and invalid action. Defaults to None. It aslo correspind to the states right before s_f and aftereards
            next_states (States, optional): States object with `uni-dimensional` batch_shape, representing the children of the transitions. Defaults to None.
            is_backward (bool, optional): Whether the transitions are backward transitions (i.e. next_states is the parent of states). Defaults to False.
            log_probs (FloatTensor1D, optional): The log-probabilities of the actions. Defaults to None.(value set to 0)

        Return:
            log_rewards (FloatTensor1D,): By calling self.log_rewards,the log-rewards of the terminating transitions,
             (from states to sink states ) is computed. (using a default value like `-1` for non-terminating transitions).
        """
        self.env = env
        self.is_backward = is_backward
        self.n_trajectories=n_trajectories

        self.states = states if states is not None else \
            env.States.from_batch_shape(batch_shape=(0,))
        assert len(self.states.batch_shape) == 1

        self.actions = actions if actions is not None else \
            torch.full(size=(0,), fill_value=-1, dtype=torch.long)
        self.is_done = is_done if is_done is not None else \
            torch.full(size=(0,), fill_value=False, dtype=torch.bool)

        self.next_states = next_states if next_states is not None else \
            env.States.from_batch_shape(batch_shape=(0,))
        assert (len(self.next_states.batch_shape) == 1 and
                self.states.batch_shape == self.next_states.batch_shape)

        self.log_probs = log_probs if log_probs is not None else torch.zeros(0)
    @property
    def n_transitions(self) -> int:
        return self.states.batch_shape[0]
    @property
    def is_sink_action(self):
        return self.actions==-1
    #assert torch.all(self.actions==-1,self.states.is_sink_state)

    @property
    def is_terminating_action(self):
        return self.actions==self.env.n_actions-1
    # assert torch.all(self.actions==self.env.n_actions-1,self.is_done)

    def __len__(self) -> int:
        return self.n_transitions

    def __repr__(self):
        states_tensor = self.states.states_tensor
        next_states_tensor = self.next_states.states_tensor

        states_repr = ",\t".join([
            f"{str(state.numpy())} -> {str(next_state.numpy())}"
            for state, next_state in zip(states_tensor, next_states_tensor)
        ])
        return (
            f"Transitions(n_transitions={self.n_transitions}, "
            f"transitions={states_repr}, actions={self.actions}, "
            f"is_done={self.is_done})"
        )

    @property
    def last_states(self) -> States:
        "Get the last states, i.e. terminating states"
        return self.states[self.is_done]

    @property
    def log_rewards(self) -> FloatTensor | None:
        if self.is_backward:
            return None
        else:
            log_rewards = torch.full( (self.n_transitions,),
                                      fill_value=-1.0, dtype=torch.float,device=self.states.device)
            log_rewards[self.is_done] = self.env.log_reward(self.last_states)
            return log_rewards

    @property
    def all_log_rewards(self) -> PairFloatTensor:
        """This is applicable to environments where all states are terminating.
        This function evaluates the rewards for all transitions that do not end in the sink state.
        This is useful for the Modified Detailed Balance loss.
        (In other words, all states in the DAG are connected to the sink, so all states rewards are meaningful.
        Here sink states does not mean valid states are just reached, only mean that non-action can be taken any more)"""
        if self.is_backward:
            raise NotImplementedError("Not implemented for backward transitions")
        is_sink_state = self.next_states.is_sink_state
        log_rewards = torch.full( (self.n_transitions, 2),
                                  fill_value=-1.0,dtype=torch.float,device=self.states.device)
        log_rewards[~is_sink_state, 0] = self.env.log_reward( self.states[~is_sink_state])
        log_rewards[~is_sink_state, 1] = self.env.log_reward( self.next_states[~is_sink_state])
        return log_rewards

    def __getitem__(self, index: int | Sequence[int]) -> Transitions:
        "Access particular transitions of the batch."
        if isinstance(index, int):
            index = [index]
        return Transitions(
            env=self.env,
            states=self.states[index],
            actions= self.actions[index],
            is_done=self.is_done[index],
            next_states=self.next_states[index],
            is_backward=self.is_backward
        )

    def extend(self, other: Transitions) -> None:
        "Extend the Transitions object with another Transitions object."
        self.states.extend(other.states)
        self.actions = torch.cat((self.actions, other.actions), dim=0)
        self.is_done = torch.cat((self.is_done, other.is_done), dim=0)
        self.next_states.extend(other.next_states)
        self.log_probs = torch.cat((self.log_probs, other.log_probs), dim=0)
    def to_device(self,device):
        for key, val in self.__dict__.items():
            if isinstance(val, Container):
                self.__getattribute__(key).to_device(device)
            elif isinstance(val, torch.Tensor):
                self.__setattr__(key, self.__getattribute__(key).to(device))

