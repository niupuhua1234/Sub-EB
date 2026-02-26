from abc import ABC #ABC is Helper class in abc that provides a standard way to create an ABC using inheritance.
from typing import Literal, Optional

import torch
from torchtyping import TensorType

from src.gfn.containers import States
from src.gfn.envs import Env
from src.gfn.modules import GFNModule, NeuralNet,Uniform
# Typing
OutputTensor = TensorType["batch_shape", "output_dim", float]


class FunctionEstimator(ABC):
    """Base class for function estimators."""

    def __init__(self,
                 env: Env,
                 module: Optional[GFNModule] = None,
                 output_dim: Optional[int] = None,
                 module_name: Optional[Literal["Transformer","NeuralNet","Uniform"]] = None,
                 **nn_kwargs
                 ) -> None:
        """Either module or (module_name, output_dim) must be provided.

        Args:
            env (Env): the environment.
            module (Optional[GFNModule], optional): The module to use. Defaults to None.
            output_dim (Optional[int], optional): Used only if module is None. Defines the output dimension of the module. Defaults to None.
            module_name (Optional[Literal[NeuralNet, Uniform, Tabular, Zero]], optional): Used only if module is None. What module to use. Defaults to None.
            **nn_kwargs: Keyword arguments to pass to the module, when module_name is NeuralNet.
        """

        self.env = env
        if module is None:
            assert module_name is not None and output_dim is not None
            if module_name == "NeuralNet":
                assert len(env.preprocessor.output_shape) == 1
                input_dim = env.preprocessor.output_shape[0]
                module = NeuralNet(input_dim=input_dim,
                                   output_dim=output_dim,
                                   **nn_kwargs
                                   )
            elif module_name == "Uniform":
                module = Uniform(output_dim=output_dim)
            else:
                raise ValueError(f"Unknown module_name {module_name}")
        self.module = module
        self.preprocessor = env.preprocessor

    def __call__(self, states: States) -> OutputTensor:
        return self.module(self.preprocessor(states))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.module})"

    def named_parameters(self) -> dict:
        return dict(self.module.named_parameters())

    def parameters(self) -> list:
        return list(self.module.parameters())

    def load_state_dict(self, state_dict: dict):
        self.module.load_state_dict(state_dict)

# Edge/State Flow estimator
class LogEdgeFlowEstimator(FunctionEstimator):
    """
    Container for estimator  :math:`(s \Rightarrow s') \mapsto \log F(s \Rightarrow s')`.
    The way it's coded is a function :math:`s \mapsto \log F (s \Rightarrow (s + a))_{a \in \mathbb{A}}`
    ,where `s+a` is the state obtained by performing action `a` in state `s`.
    """
    def __init__(self,env,module=None,module_name=None,**nn_kwargs,) -> None:
        if module is not None:
            assert module.output_dim == env.n_actions
        super().__init__(env,module=module,output_dim=env.n_actions,module_name=module_name,**nn_kwargs)
    def __call__(self, states: States) -> OutputTensor:
        out = super().__call__(states)
        return out

class LogStateFlowEstimator(FunctionEstimator):
    r"""Container for estimators :math:`s \mapsto \log F(s)`."""
    def __init__(self,env,module=None,module_name=None,**nn_kwargs):
        if module is not None:
            assert module.output_dim == 1
        super().__init__(env, module=module,output_dim=1,module_name=module_name,**nn_kwargs)
    def __call__(self, states: States) -> OutputTensor:
        out = super().__call__(states)
        return out

# Forward/Backward Prob estimator
class LogitPFEstimator(FunctionEstimator):
    r"""
    Container for estimators :math:`s \mapsto u(s + a \mid s)_{a \in \mathbb{A}}` ,
    such that :math:`P_F(s + a \mid s) = \frac{e^{u(s + a \mid s)}}{\sum_{a' \in \mathbb{A}} e^{u(s + a' \mid s)}}`.
    """
    def __init__(self,env,module=None,module_name=None,**nn_kwargs):
        if module is not None:
            assert module.output_dim == env.n_actions
        super().__init__(env, module=module, output_dim=env.n_actions,module_name=module_name,**nn_kwargs)

    def __call__(self, states: States) -> OutputTensor:
        out = super().__call__(states)
        return out

class LogitPBEstimator(FunctionEstimator):
    r"""Container for estimators :math:`s' \mapsto u(s' - a \mid s')_{a \in \mathbb{A}}` ,
    such that :math:`P_B(s' - a \mid s') = \frac{e^{u(s' - a \mid s')}}{\sum_{a' \in \mathbb{A}} e^{u(s' - a' \mid s')}}`."""

    def __init__(self,env,module=None,module_name=None,**nn_kwargs):
        if module is not None:
            assert module.output_dim == env.n_bctions
        super().__init__(env, module=module, output_dim=env.n_bctions,module_name=module_name,**nn_kwargs)
    def __call__(self, states: States) -> OutputTensor:
        out = super().__call__(states)
        return out

class LogZEstimator:
    r"""Container for the estimator `\log Z`."""
    def __init__(self, tensor: TensorType[0, float]) -> None:
        self.tensor = tensor
        assert self.tensor.shape == ()
        self.tensor.requires_grad = True

    def __repr__(self) -> str:
        return str(self.tensor.item())

    def named_parameters(self) -> dict:
        return {"logZ": self.tensor}

    def parameters(self) -> list:
        return [self.tensor]

    def load_state_dict(self, state_dict: dict):
        self.tensor = state_dict["logZ"]