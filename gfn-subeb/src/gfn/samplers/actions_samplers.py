from abc import ABC, abstractmethod
from typing import Tuple
import pdb
import torch
from torch.distributions import Categorical
from torchtyping import TensorType

from src.gfn.containers.states import States
from src.gfn.estimators import LogEdgeFlowEstimator, LogitPBEstimator, LogitPFEstimator,LogStateFlowEstimator

# Typing
Tensor2D = TensorType["batch_size", "n_actions"]
Tensor2D2 = TensorType["batch_size", "n_steps"]# only check 2 dimesions
Tensor1D = TensorType["batch_size", torch.long] # check 1 dimension and its type


class ActionsSampler(ABC):
    """
    Base class for action sampling methods. (forward)
    """

    @abstractmethod
    def sample(self, states: States) -> Tuple[Tensor1D, Tensor1D]:
        """
        Args:
            states (States): A batch of states.

        Returns:
            Tuple[Tensor[batch_size], Tensor[batch_size]]: A tuple of tensors containing the log probabilities of the sampled actions, and the sampled actions.
        """
        pass
    def action_prob_gather(self,probs,actions):
        return torch.gather(probs,dim=-1,index=actions.unsqueeze(-1)).squeeze(-1)


class BackwardActionsSampler(ActionsSampler):
    """
    Base class for backward action sampling methods.
    """
    pass


class DiscreteActionsSampler(ActionsSampler):
    """
    For Discrete environments.(forward)
    """

    def __init__(self,
                 estimator: LogitPFEstimator | LogEdgeFlowEstimator|LogitPBEstimator,
                 temperature: float = 1.0,
                 sf_bias: float = 0.0,
                 epsilon: float = 0.0,
                 ) -> None:
        """Implements a method that samples actions from any given batch of states.

        Args:
            temperature (float, optional): scalar to divide the logits by before softmax. Defaults to 1.0.
            sf_bias (float, optional): scalar to subtract from the exit action logit before dividing by temperature. Defaults to 0.0.
            epsilon (float, optional): with probability epsilon, a random action is chosen. Defaults to 0.0.
        """
        self.estimator = estimator
        self.temperature = temperature
        self.sf_bias = sf_bias
        self.epsilon = epsilon

    def get_logits(self, states: States,fill_value=-float('inf')) -> Tensor2D:
        """Transforms the raw logits by masking illegal actions.  0 probs is assigned -inf logits

        Raises:
            ValueError: if one of the resulting logits is NaN.

        Returns:
            Tensor2D: A 2D tensor of shape (batch_size, n_actions) containing the transformed logits.
        """
        logits = self.estimator(states)
        if torch.any(torch.all(torch.isnan(logits), -1)):
            raise ValueError("NaNs in estimator")    # 0 masking to be log(0)=-inf??
        logits[~states.forward_masks] = fill_value   # -float('inf') softmax -> 0.   logsoftmax -> -inf
        return logits

    def get_probs(self,states: States,actions:Tensor1D=None) -> Tensor2D|Tensor1D:
        """
        Returns:
            The probabilities of each action in each state in the batch.
        """
        with  torch.no_grad():
            logits = self.get_logits(states)
            logits[..., -1] -= self.sf_bias
            probs = torch.softmax(logits / self.temperature, dim=-1)  # softmax[(P_logit-bias)/T]  [-inf, 0.2]softmax-> [0,1]
            if actions is None:
                # when the element of  a vector is all inf softmax will return Nan, this means that all actions are maksed this is in s_f.
                if torch.any(torch.all(torch.isnan(probs), -1)):
                    raise ValueError("No terminating action or further action is allowed ")
            else:
                probs = self.action_prob_gather(probs, actions)
        return probs


    def sample(self, states: States) -> Tuple[Tensor1D, Tensor1D]:
        probs = self.get_probs(states)
        dist = Categorical(probs=probs)
        if self.epsilon>0.:
            #unifrom distribution
            masks = states.forward_masks.float()
            uniform_prob =  masks/masks.sum(dim=-1, keepdim=True)
            uniform_dist = Categorical(probs=uniform_prob)

            # with 1-epsilon to use Pf probs and epsilon to use unifrom probs
            choice = torch.bernoulli(torch.ones(*probs.shape[:-1],device=states.device) * self.epsilon).bool()
            actions = torch.where(choice,uniform_dist.sample(),dist.sample())
            actions_log_probs = torch.where(choice,uniform_dist.log_prob(actions),dist.log_prob(actions))

            while torch.any(self.action_prob_gather(dist.probs,actions)==0.):  # some impossible event with probability < e-10 happens resample !
                actions = torch.where(choice, uniform_dist.sample(), dist.sample())
                actions_log_probs = torch.where(choice, uniform_dist.log_prob(actions), dist.log_prob(actions))
        else:
            actions = dist.sample()
            actions_log_probs = dist.log_prob(actions)
            while torch.any(self.action_prob_gather(dist.probs,actions)==0.):   #while torch.any(actions_log_probs.abs() > 15):
                actions = dist.sample()
                actions_log_probs = dist.log_prob(actions)
        return actions_log_probs, actions

class BackwardDiscreteActionsSampler(DiscreteActionsSampler, BackwardActionsSampler):
    """
    For sampling backward actions in discrete environments.
    """

    def __init__(self,
                 estimator: LogitPBEstimator,
                 temperature: float = 1.0,
                 epsilon: float = 0.0,
                 ) -> None:
        """s_f is not biased in the backward sampler."""
        super().__init__(estimator,
                         temperature=temperature,
                         sf_bias=0.0,
                         epsilon=epsilon)

    def get_logits(self, states: States,fill_value=-float('inf')) -> Tensor2D:
        logits =  self.estimator(states)
        if torch.any(torch.all(torch.isnan(logits), -1)):
            raise ValueError("NaNs in estimator")
        logits[~states.backward_masks] = fill_value
        return logits

    def get_probs(self, states: States, actions: Tensor1D =None) -> Tensor2D|Tensor1D:
        """
        Unlike forward pass there is always a non-masked valid action, the terminating action,
        In bakcward pass  reaching (0,0),  any further actions are masked and the probability is NaN.
        """
        logits = self.get_logits(states)
        with torch.no_grad():
            probs = torch.softmax(logits / self.temperature, dim=-1)
            if actions is None:
                if torch.any(torch.all(torch.isnan(probs), -1)):
                    raise ValueError("No terminating action or further action is allowed ")
            else:
                probs = self.action_prob_gather(probs, actions)
        return probs

    def sample(self, states: States) -> Tuple[Tensor1D, Tensor1D]:
        probs = self.get_probs(states)
        dist = Categorical(probs=probs)
        actions = dist.sample()
        actions_log_probs = dist.log_prob(actions)
        while torch.any(self.action_prob_gather(dist.probs,actions)==0.):  # some impossible event with probability < e-10 happens resample !
            actions=dist.sample()
            actions_log_probs=dist.log_prob(actions)
        return actions_log_probs, actions
