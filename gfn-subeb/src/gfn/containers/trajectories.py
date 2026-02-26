from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from src.gfn.envs import Env
if TYPE_CHECKING:
    from src.gfn.containers.states import States

import torch
from torchtyping import TensorType

from src.gfn.containers.base import Container
from src.gfn.containers.transitions import Transitions

# Typing  --- n_transitions is an int
Tensor2D = TensorType["max_length", "n_trajectories", torch.long]
FloatTensor2D = TensorType["max_length", "n_trajectories", torch.float]
Tensor2D2 = TensorType["n_trajectories", "shape"]
Tensor1D = TensorType["n_trajectories", torch.long]
FloatTensor1D = TensorType["n_trajectories", torch.float]


class Trajectories(Container):
    def __init__(self,
                 env: Env,
                 states:       States | None = None,
                 actions:      Tensor2D | None = None,
                 when_is_done: Tensor1D | None = None,
                 is_backward:  bool |None  = False,
                 log_probs: FloatTensor2D | None = None,
                 log_rewards: FloatTensor1D | None = None,
                 ) -> None:
        """
        Container for complete trajectories (starting in s_0 and ending in s_f).
        `Trajectories` are represented via a `States` object where ``states.tensor`` is of bi-dimensional batch shape.
        The first dimension represents the time step [0,T+1], the second dimension represents the trajectory index [0,N-1].

        Because different trajectories may have different lengths, shorter trajectories are padded with
        the tensor representation of the terminal state (s_f or s_0 depending on the direction of the trajectory), and
        actions is appended with -1's.

        If states is None, then the states are initialized to an empty States object, that can be populated on the fly.

        The actions are represented as a `two-dimensional` tensor with the first dimension representing the time step [0,T]
        and the second dimension representing the trajectory index.

        e.g.
        Forward trajectory of length=T+1. It contains T+2 states ( +1 for sf), and T+1 transitions.
                                                    s_0->s1->......s_T ->sf.
                                                    a_0->a1->......a_T.
        Args:
            env (Env): The environment in which the trajectories are defined.
            states (States, optional)            : The states of the trajectories                             Defaults to None
                                                  Its length is 1 bigger than actions for s_f forward or s_0 backward .
            actions (Tensor2D, optional)         : The actions of the trajectories                            Defaults to None.
            when_is_done (Tensor1D, optional)    : The time step at which each trajectory ends. ∈[1,T+1]
            is_backward (bool, optional)         : Whether the trajectories are backward or forward.          Defaults to False.
            log_probs (FloatTensor2D, optional)  : The log probabilities of the trajectories' actions.        Defaults to None.
            log_rewards(FloatTensor1D, optional) : THe log rewards of the trajectories' terminating states.   Defaults to None

        Returns:
            log_rewards (FloatTensor1D): The log_rewards of the trajectories, (Rewards of the  states before sink states).
             ``env.log_reward`` is used to compute the rewards,at each call of ``self.log_rewards``
        """
        self.env = env
        self.is_backward = is_backward

        self.states = (states if states is not None else env.States.from_batch_shape(batch_shape=(0, 0)))
        assert len(self.states.batch_shape) == 2 #(time batch)
        self.actions = (actions if actions is not None else torch.full(size=(0, 0), fill_value=-1, dtype=torch.long,device=self.states.device))
        self.log_probs = ( log_probs if log_probs is not None else torch.full(size=(0, 0), fill_value=0, dtype=torch.float,device=self.states.device))
        self.when_is_done = (when_is_done if when_is_done is not None else torch.full(size=(0,), fill_value=-1, dtype=torch.long,device=self.states.device))
        self._log_rewards = log_rewards

    def to_device(self,device):
        for key, val in self.__dict__.items():
            if isinstance(val, Container):
                self.__getattribute__(key).to_device(device)
            elif isinstance(val, torch.Tensor):
                self.__setattr__(key, self.__getattribute__(key).to(device))

    def __repr__(self) -> str:    # give a description of the trajectory object
        states = self.states.states_tensor.transpose(0, 1)  # (time batch)->(batch,time)
        assert states.ndim == 3
        trajectories_representation = ""
        for traj in states[:10]:    #show at most ten trajectories
            one_traj_repr = []

            for step in traj:
                one_traj_repr.append(str(step.numpy()))
                if step.equal(self.env.s0 if self.is_backward else self.env.sf): break

            trajectories_representation += "-> ".join(one_traj_repr) + "\n"

        return (
            f"Trajectories(n_trajectories={self.n_trajectories}, "
            f"max_length={self.max_length}, "
            f"First 10 trajectories: "
            f"states=\n{trajectories_representation}, "
            f"actions=\n{self.actions.transpose(0, 1)[:10].numpy()}, "
            f"when_is_done={self.when_is_done[:10].numpy()})"
        )

    @property
    def n_trajectories(self) -> int:
        return self.states.batch_shape[1]

    def __len__(self) -> int:
        return self.n_trajectories

    @property
    def max_length(self) -> int:
        if len(self) == 0:
            return 0
        #If when_is_done is given assert self.when_is_done.max()==self.actions.shape[0]
        return self.actions.shape[0] #note the length doesnot cound the final s_f or s_0 in backward

    @property
    def last_states(self) -> States:
        return self.states[self.when_is_done - 1, torch.arange(self.n_trajectories)] # select s of s->s_f
        # Note we need explict specifiy all the index [i..],[j...] pair  to only query the last states
    @property
    def intermediary_states(self)->States:
        states=self.states
        return states[~states.is_sink_state & ~states.is_initial_state]
    @property
    def is_sink_action(self):
        return self.env.is_sink_actions(self.actions)
    @property
    def is_terminating_action(self):
        return self.env.is_exit_actions(self.actions)

    @property
    def log_rewards(self) -> FloatTensor1D | None:
        if self._log_rewards is None:
            self._log_rewards=self.env.log_reward(self.states[0])  if self.is_backward \
                else self.env.log_reward(self.last_states)
        assert self._log_rewards.shape == (self.n_trajectories,)
        return self._log_rewards

    def __setitem__(self, index: int | Sequence[int], value: 'Trajectories'):
        "Assigns a subset of trajectories at given index with the provided `value` Trajectories object."
        if isinstance(index, int):index=[index]
        # Check length compatibility
        assert (self.when_is_done[index]==value.when_is_done).all().item(), "Mismatch in lengths trajectories to assign"

        new_max_length = value.when_is_done.max().item()

        self.states[:1 + new_max_length, index] = value.states[:1 + new_max_length]
        self.actions[:new_max_length, index]    = value.actions[:new_max_length]
        self.when_is_done[index]                = value.when_is_done
        self.log_probs[:new_max_length, index]  = value.log_probs[:new_max_length]

        if self._log_rewards is not None and value._log_rewards is not None:
            self._log_rewards[index] = value._log_rewards

    def __getitem__(self, index: int | Sequence[int]) -> Trajectories:
        "Returns a subset of the `n_trajectories` trajectories."
        if isinstance(index, int):  index = [index]
        # we don't need to print each traj with the max-length of all stored trajs.
        new_max_length = self.when_is_done[index].max().item() if len(self.when_is_done[index]) > 0 else 0
        #note the max_length does not count the s_f
        return Trajectories(
            env=           self.env,
            states=        self.states[:1 + new_max_length, index],
            actions=       self.actions[:new_max_length,    index],
            when_is_done=  self.when_is_done[index],
            is_backward=   self.is_backward,
            log_probs=     self.log_probs[:new_max_length,  index],
            log_rewards=   self._log_rewards[index] if self._log_rewards is not None else None
        )

    def extend(self, other: Trajectories) -> None:
        """
        Extend the trajectories with another set of trajectories.
        ('Merge a batch of trajs with  another batch??')
        """
        self.extend_actions(required_first_dim=max(self.max_length, other.max_length))  # make the max-length to be consistent between two batches
        other.extend_actions(required_first_dim=max(self.max_length, other.max_length)) # so that the two batch can be combine along batch dim.

        self.states.extend(other.states)
        self.actions = torch.cat((self.actions, other.actions), dim=1)
        self.when_is_done = torch.cat((self.when_is_done, other.when_is_done), dim=0)
        self.log_probs = torch.cat((self.log_probs, other.log_probs), dim=1)
        self._log_rewards = torch.cat((self._log_rewards, other._log_rewards), dim=0)  \
            if self._log_rewards is not None and other._log_rewards is not None else None


    def extend_actions(self, required_first_dim: int) -> None:
        """
        Extends the `actions and log_probs` along the first dimension by adding `-1s` and `0s` as necessary.
        This is useful for extending trajectories of different lengths.
        """
        if self.max_length >= required_first_dim:
            return
        action_padding=required_first_dim - self.actions.shape[0]
        logp_padding=required_first_dim - self.log_probs.shape[0]
        self.actions = torch.cat( (self.actions,torch.full(size=(action_padding,self.n_trajectories,),
                                                           fill_value=-1,dtype=torch.long,device=self.actions.device)),dim=0)
        self.log_probs = torch.cat((self.log_probs,torch.full(size=( logp_padding,self.n_trajectories,),
                                                              fill_value=0,dtype=torch.float,device=self.log_probs.device),),dim=0)

    def to_transitions(self) -> Transitions:
        """
        Returns a `Transitions` object from the trajectories
        """

        current_states = self.states[:-1][self.actions != -1]
        next_states = self.states[1:][self.actions != -1]    # filter-out padding psedo-sink states with -1 action values
        is_done = ( next_states.is_sink_state if not self.is_backward else next_states.is_initial_state) # `n_traj`

        return Transitions(
            env=self.env,
            states=current_states,
            actions= self.actions[self.actions != -1] ,
            is_done=is_done,
            next_states=next_states,
            is_backward=self.is_backward,
            log_probs=self.log_probs[self.actions != -1],
            n_trajectories=self.n_trajectories
        )

    def to_states(self) -> States:
        """Returns a `States` object from the trajectories, containing all states in the trajectories"""
        states = self.states.flatten()
        return states[~states.is_sink_state]

    def revert_backward_trajectories(self) -> Trajectories:
        """
        Return a forward trajectories from a backward trajectories
        In the forward τ_f,  s_f is sink state and not counted but s_0 is
        In the backward τ_b, s_0 is the sink state
        So the correspond τ_f is one step longer than the τ_b
                                                    Backward trajectory
                                                    s_0 ← s_1 ←.......... ← s_T-1 ← s_T
                                                    ___   b_1 ← b_2.......← b_T-1 ← b_T
                                                              ↓
                                                    a_0 ← a_1 ←.....a_T-2 ← a_T-1

                                                    Forward trajectory
                                                    s_0 → s_1 →.......... → s_T-1 → s_T → sf .
                                                    a_0 → a_1 →...→ a_T-2 → a_T-1 → (a_f)  ___
                                                                            logprob of a_f is missing note!!
        Besides a_1<-a_2<-a_3....a_T  in τ_B is changed into  a_0->a_1->a_2.....a_T-> a_f   (a_f : s->s_f)
        """
        assert self.is_backward

        new_actions=torch.full((self.max_length+1,self.n_trajectories),-1, dtype=torch.long,device=self.actions.device)
        new_logps=torch.full((self.max_length+1,self.n_trajectories),0., dtype=torch.float,device=self.log_probs.device)

        new_states = self.env.sf.repeat(self.max_length+2, self.n_trajectories, 1) #+1 for s_0 in backward  traj + 1 for s_f in convetred forward traj
        new_when_is_done = self.when_is_done + 1                                   #+1 as s_0 is counted in forward traj
        for i in range(self.n_trajectories):
            new_actions[: self.when_is_done[i], i] = self.env.bction2action(self.states[ : self.when_is_done[i], i],
                                                                            self.actions[: self.when_is_done[i], i]).flip(0)
            new_logps  [: self.when_is_done[i], i] = self.log_probs[: self.when_is_done[i],i].flip(0)
            new_actions[self.when_is_done[i], i] = (self.env.n_actions - 1)                  # add action  s->s_f with logprob=0
            new_states [: self.when_is_done[i] + 1, i] = self.states.states_tensor[: self.when_is_done[i] + 1, i].flip(0) #+1 to include s_0

        new_states = self.env.States(new_states)
        new_states.to_device(self.states.device)
        return Trajectories(
            env=self.env,
            states=new_states,
            actions=new_actions,
            when_is_done=new_when_is_done,
            log_probs=new_logps,
            is_backward=False,
        )

