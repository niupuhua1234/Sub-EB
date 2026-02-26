import numpy as np
import math

from scipy.special import gammaln
from abc import ABC, abstractmethod

class BasePrior(ABC):
    """Base class for the prior over graphs p(G).

    Any subclass of `BasePrior` must return the contribution of log p(G) for a
    given variable with `num_parents` parents. We assume that the prior is modular.

    Parameters
    ----------
    num_nodes : int (optional)
        The number of nodes in the graph. If not specified, this gets
        populated inside the scorer class.
    """

    def __init__(self, num_nodes=None):
        self._num_nodes = num_nodes
        self._log_prior = None

    def __call__(self, num_parents):
        if self.log_prior is None:
            raise ValueError("log_prior has not been initialized.")
        return self.log_prior[num_parents]

    @property
    @abstractmethod
    def log_prior(self):
        pass

    @property
    def num_nodes(self):
        if self._num_nodes is None:
            raise RuntimeError('The number of nodes is not defined.')
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, value):
        self._num_nodes = value


class UniformPrior(BasePrior):
    @property
    def log_prior(self):
        if self._log_prior is None:
            self._log_prior = np.zeros((self.num_nodes,))
        return self._log_prior

class ErdosRenyiPrior(BasePrior):
    def __init__(self, num_nodes=None, num_edges_per_node=1.):
        super().__init__(num_nodes)
        self.num_edges_per_node = num_edges_per_node

    @property
    def log_prior(self):
        if self._log_prior is None:
            num_edges = self.num_nodes * self.num_edges_per_node  # Default value
            p = num_edges / ((self.num_nodes * (self.num_nodes - 1)) // 2)
            all_parents = np.arange(self.num_nodes)
            self._log_prior = (all_parents * math.log(p)
                + (self.num_nodes - all_parents - 1) * math.log1p(-p))
        return self._log_prior


class EdgePrior(BasePrior):
    def __init__(self, num_nodes=None, beta=1.):
        super().__init__(num_nodes)
        self.beta = beta

    @property
    def log_prior(self):
        if self._log_prior is None:
            self._log_prior = np.arange(self.num_nodes) * math.log(self.beta)
        return self._log_prior


class FairPrior(BasePrior):
    @property
    def log_prior(self):
        if self._log_prior is None:
            all_parents = np.arange(self.num_nodes)
            self._log_prior = (
                - gammaln(self.num_nodes + 1)
                + gammaln(self.num_nodes - all_parents + 1)
                + gammaln(all_parents + 1)
            )
        return self._log_prior