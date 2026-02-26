from __future__ import annotations
from abc import ABC, abstractmethod
import os
import torch
import numpy as np
from torchtyping import TensorType
from src.gfn.containers.states import States
from src.gfn.containers.transitions import Transitions
from src.gfn.containers.trajectories import Trajectories
from src.gfn.envs import Env

TensorFloat = TensorType["n_trajectories", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]
class ReplayBuffer(ABC):
    def __init__(
        self,
        env: Env,
        capacity: int = 1000,
    ):
        self.env = env
        self.capacity = capacity
        self._is_full = False
        self._index = 0
    def __repr__(self):
        return f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} training samples)"
    def __len__(self):
        return self.capacity if self._is_full else self._index

class Replay_x(ReplayBuffer):  # replay of terminating states
    def __init__(self,
                 env: Env,
                 capacity: int = int(1e10), ):
        super().__init__(env,capacity)
        self.x_states = torch.LongTensor(0, *env.s0.shape).to(env.device)
        #self.x_index   = torch.LongTensor(0)
        self.x_rewards = torch.FloatTensor(0).to(env.device)

    def __repr__(self):
        return f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} terminating states)"
    @property
    def unique_states_rewards(self):
        states_np = self.x_states.to('cpu').numpy()
        rewards_np = self.x_rewards.to('cpu').numpy()
        uniq, idx = np.unique(states_np, axis=0, return_index=True)
        unique_states = torch.from_numpy(uniq)
        unique_rewards = torch.from_numpy(rewards_np[idx])
        return unique_states,unique_rewards

    def add(self, x_states:StatesTensor, rewards: TensorFloat):
        x_states= x_states if  x_states.ndim >1 else x_states.unsqueeze(0)
        to_add = len(x_states)
        self._is_full |= self._index + to_add >= self.capacity
        self._index = (self._index + to_add) % self.capacity
        #
        self.x_states = torch.cat((self.x_states, x_states))
        self.x_rewards = torch.cat((self.x_rewards, rewards))
        self.x_states = self.x_states[-self.capacity:]
        self.x_rewards = self.x_rewards[-self.capacity:]

    def save(self, path: str,index:str) -> None:
        "Saves the container to a file"
        for key, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                torch.save(val, os.path.join(path, index+key+".pt"))

    def load(self, path: str,index:str) -> None:
        "Loads the container from a file, overwriting the current container"
        for key, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                self.__dict__[key] = torch.load(os.path.join(path, index+key+".pt"))
        self._index = len(self.x_states)

    def sample(self, n_trajectories: int):
        indices=torch.randint(0,len(self),size=(n_trajectories,))
        return self.x_states[indices]#,self.x_rewards[indices]

    def sample_biased(self, n_samples: int):
        """Samples a subset of the container:
           50% from top 10% rewards, 50% from bottom 90%."""
        if len(self) < 10:
            return self.x_states[:0],self.x_rewards[:0]
        threshold = torch.quantile(self.x_rewards, 0.9)

        top_indices = torch.where(self.x_rewards >= threshold)[0]
        bottom_indices = torch.where(self.x_rewards < threshold)[0]

        top_sampled    = top_indices[torch.randint(len(top_indices), (n_samples // 2,))]
        bottom_sampled = bottom_indices[torch.randint(len(bottom_indices), (n_samples- n_samples // 2,))]

        all_sampled = torch.cat([top_sampled, bottom_sampled])
        return self.x_states[all_sampled ]#,self.x_rewards[all_sampled ]

class  Replay_Traj(ReplayBuffer):
    def __init__(
        self,
        env: Env,
        capacity: int = 1000,
    ):
        super().__init__(env,capacity)
        self.training_objects = Trajectories(env)
    def __repr__(self):
        return f"ReplayBuffer(capacity={self.capacity}, containing {len(self)} trajectories)"
    def add(self, training_objects: Trajectories):
        to_add = len(training_objects)
        self._is_full |= self._index + to_add >= self.capacity
        self._index = (self._index + to_add) % self.capacity

        self.training_objects.extend(training_objects)
        self.training_objects = self.training_objects[-self.capacity :]

    def sample(self, n_trajectories: int) -> Trajectories:
        indices=torch.randint(0,len(self),size=(n_trajectories,))
        return self.training_objects[indices]

    def save(self, directory: str,index:str):
        self.training_objects.save(os.path.join(directory, index+"training_objects"))

    def load(self, directory: str,index:str):
        self.training_objects.load(os.path.join(directory, index+"training_objects"))
        self._index = len(self.training_objects)
