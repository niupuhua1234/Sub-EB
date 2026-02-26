from abc import ABC, abstractmethod
from typing import Callable, Tuple

from torchtyping import TensorType

from src.gfn.containers import States

# Typing
OutputTensor = TensorType["batch_shape", "dim_in"]


class Preprocessor(ABC):
    """
    Base class for Preprocessors. The goal is to transform tensors representing raw states
    to tensors that can be used as input to neural networks.
    """

    name: str = "Preprocessor"

    def __init__(self, output_shape: Tuple[int,...]) -> None:
        self.output_shape = output_shape

    @abstractmethod
    def preprocess(self, states: States) -> OutputTensor:
        pass

    def __call__(self, states: States) -> OutputTensor:
        return self.preprocess(states)

    def __repr__(self):
        return f"{self.name}, output_shape={self.output_shape}"




