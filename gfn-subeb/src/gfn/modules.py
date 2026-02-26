from abc import ABC, abstractmethod
from typing import Literal, Optional
import torch
import torch.nn as nn
from torch.nn import Transformer,TransformerEncoder,TransformerEncoderLayer

from torchtyping import TensorType
# Typing

InputTensor = TensorType["batch_shape", "input_shape", float]
OutputTensor = TensorType["batch_shape", "output_dim", float]

class GFNModule(ABC):
    """Abstract Base Class for all functions/approximators/estimators used.
    Each module takes a preprocessed tensor as input, and outputs a tensor of logits,
    or log flows. The input dimension of the module (e.g. Neural network), is deduced
    from the environment's preprocessor's output dimension"""

    def named_parameters(self) -> dict:
        """Returns a dictionary of all (learnable) parameters of the module. Not needed
        for NeuralNet modules, given that those inherit this function from nn.Module"""
        return {}

    def parameters(self) -> list:
        """Returns a dictionary of all (learnable) parameters of the module. Not needed
        for NeuralNet modules, given that those inherit this function from nn.Module"""
        return []

    @abstractmethod
    def __call__(self, preprocessed_states: InputTensor) -> OutputTensor:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        pass

class NeuralNet(nn.Module, GFNModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = 256,
        n_hidden_layers: Optional[int] = 2,
    ):
        """Implements a basic MLP.

        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            torso (Optional[nn.Module], optional): If provided, this module will be used as the torso of the network (i.e. all layers except last layer). Defaults to None.
        """
        super().__init__()
        self.output_dim = output_dim
        self.torso = [nn.Linear(input_dim, hidden_dim),  nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            self.torso.append(nn.Linear(hidden_dim, hidden_dim))
            self.torso.append( nn.ReLU())
        self.torso = nn.Sequential(*self.torso)
        self.last_layer = nn.Linear(hidden_dim, output_dim)
        self.device = None

    def forward(self, preprocessed_states: InputTensor) -> OutputTensor:
        #if self.device is None:
        self.device = preprocessed_states.device
        self.to(self.device)
        logits = self.torso(preprocessed_states)
        logits = self.last_layer(logits)
        return logits

class Tabular(GFNModule):
    def __init__(self, n_states: int, output_dim: int) -> None:
        """Implements a tabular function approximator. Only compatible with the EnumPreprocessor.
        Args:
            n_states (int): Number of states in the environment.
            output_dim (int): Output dimension.
        """
        self.output_dim = output_dim
        self.logits = torch.zeros((n_states, output_dim),dtype=torch.float,requires_grad=True)
        self.device = None

    def __call__(self, preprocessed_states: InputTensor) -> OutputTensor:
        if self.device is None:
            self.device = preprocessed_states.device
            self.logits = self.logits.to(self.device)
        assert preprocessed_states.dtype == torch.long
        outputs = self.logits[preprocessed_states.squeeze(-1)]
        return outputs

    def named_parameters(self) -> dict:
        return {"logits": self.logits}

    def load_state_dict(self, state_dict: dict):
        self.logits = state_dict["logits"]


class ZeroGFNModule(GFNModule):
    def __init__(self, output_dim: int) -> None:
        """Implements a zero function approximator, i.e. a function that always outputs 0.

        Args:
            output_dim (int): Output dimension.
        """
        self.output_dim = output_dim

    def __call__(self, preprocessed_states: InputTensor) -> OutputTensor:
        out = torch.zeros(*preprocessed_states.shape[:-1], self.output_dim).to(preprocessed_states.device)
        return out

    def load_state_dict(self, state_dict: dict):
        pass


class Uniform(ZeroGFNModule):
    """Use this module for uniform policies for example. This is because logits = 0 is equivalent to uniform policy"""
    pass