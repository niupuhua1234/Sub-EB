from typing import Callable,Tuple

import torch
from einops import rearrange
from torch.nn.functional import one_hot
from torch.nn import  Embedding
from torchtyping import TensorType

from src.gfn.containers.states import States,StatesTensor
from src.gfn.envs.preprocessors.base import Preprocessor

# Typing
OutputTensor = TensorType["batch_shape"]

class DictPreprocessor(Preprocessor):
    name = "dictionary"

    def __init__(
            self,
            n_dim,
            embed_dim=128,
    ):
        super().__init__(output_shape=(embed_dim*2,)) # edege *2
        self.embed_dim=embed_dim
        self.n_dim=n_dim

        indices = torch.arange(self.n_dim ** 2)
        # Create the edges as pairs of indices (source, target)
        sources, targets = torch.div(indices,self.n_dim,rounding_mode='floor'),\
                           torch.fmod(indices, self.n_dim)  # s: 0 t [0,10]  s=1 t=[0,10].....
        edges = torch.stack((sources,self.n_dim + targets), dim=-1)  # +num_variables note
        # Embedding of the edges
        embed = Embedding(num_embeddings=2 * self.n_dim, embedding_dim= embed_dim)
        with torch.no_grad():
            self.edge_dict= embed(edges).reshape(self.n_dim ** 2,-1)

    def preprocess(self, states: States) -> OutputTensor:
        return states.states_tensor.float()


class IdentityPreprocessor(Preprocessor):
    """Simple preprocessor applicable to environments with uni-dimensional states.
    This is the default preprocessor used."""

    name = "IdentityPreprocessor"

    def preprocess(self, states: States) -> OutputTensor:
        return states.states_tensor.float()

class OneHotPreprocessor(Preprocessor):
    name = "one_hot"
    def __init__(
        self,
        n_states: int,
        get_states_indices: Callable[[States], OutputTensor],
    ) -> None:
        """One Hot Preprocessor for environments with enumerable states (finite number of states).

        Args:
            n_states (int): The total number of states in the environment (not including s_f).
            get_states_indices (Callable[[States], OutputTensor]): function that returns the unique indices of the states.
        """
        super().__init__(output_shape=(n_states,))
        self.get_states_indices = get_states_indices
        self.output_dim = n_states

    def preprocess(self, states):
        state_indices = self.get_states_indices(states)
        return one_hot(state_indices, self.output_dim).float()

class KHotPreprocessor(Preprocessor):
    name = "k_hot"

    def __init__(
        self,
        height: int,
        ndim: int,
        fill_value=None
    ) -> None:
        """K Hot Preprocessor for environments with enumerable states (finite number of states) with a grid structure.

        Args:
            height (int): number of unique values per dimension.
            ndim (int): number of dimensions.
            get_states_indices (Callable[[States], OutputTensor]): function that returns the unique indices of the states.
        """
        super().__init__(output_shape=(height * ndim,))
        self.height = height
        self.ndim = ndim
        self.fill_value=fill_value

    def preprocess(self, states)->OutputTensor:
        assert (states.states_tensor.dtype == torch.long), "K Hot preprocessing only works for integer states"
        #states_tensor = states_tensor.long()
        if self.fill_value is None:
            hot = one_hot(states.states_tensor, self.height)
        else:
            index= states.states_tensor!=self.fill_value
            hot  = torch.full((*states.batch_shape,self.ndim, self.height),fill_value=0, dtype=torch.long, device=states.device)
            hot[index]  =one_hot(states.states_tensor[index],self.height)
        hot = rearrange(hot, "... a b -> ... (a b)").float() # 对最后两个dims  按[0,....] + [1,...] ->[0,...,1,...] 的方式 合并dims
        return hot