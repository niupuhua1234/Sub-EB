import math
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import torch
from scipy.linalg import block_diag
from scipy.special import gammaln
class BaseScore(ABC):
    """Base class for the scorer.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.

    prior : `BasePrior` instance
        The prior over graphs p(G).
    """
    def __init__(self, data, prior):
        self.data = data
        self.prior = prior
        self.column_names = list(data.columns)
        self.num_nodes = len(self.column_names)
        self.prior.num_nodes = self.num_nodes

    def __call__(self, graphs:torch.Tensor):
        device=graphs.device
        graphs=graphs.to('cpu').numpy()
        assert graphs.ndim==1 or graphs.ndim ==2
        if graphs.ndim<2: graphs=graphs[None,:]
        batch_size=graphs.shape[0]
        graphs = graphs.reshape(batch_size,self.num_nodes, self.num_nodes)
        scores=np.zeros(batch_size)

        for node,_ in enumerate(self.column_names):
            edge_idx = np.nonzero(graphs[:,:,node]) # (batch_idx, par_idx)
            par_idx  = np.repeat(np.arange(self.num_nodes, 2 * self.num_nodes)[None],batch_size,axis=0)
            par_idx[edge_idx[0], edge_idx[1]] = edge_idx[1]
            par_num  = np.int64(graphs[:,:,node].sum(-1))
            scores   = scores + self.batch_local_scores(node, par_idx,par_num)
        return  torch.tensor(scores,dtype=torch.float,device=device)

    def singe_call(self, graphs:torch.Tensor):
        device=graphs.device
        graphs=graphs.numpy()
        assert graphs.ndim==1 or graphs.ndim ==2
        if graphs.ndim<2: graphs=graphs[None,:]
        batch_size=graphs.shape[0]
        graphs = graphs.reshape(batch_size,self.num_nodes, self.num_nodes)
        scores=np.zeros(batch_size)

        for i,graph in enumerate(graphs):
            local_score=0.
            for node,_ in enumerate(self.column_names):
                par_idx= np.nonzero(graph[ :, node])[0]
                par_num = int(graph[ :, node].sum(-1))
                local_score = local_score + self.single_local_scores(node, par_idx, par_num)
            scores[i]=local_score
        return torch.tensor(scores,dtype=torch.float,device=device)

    @abstractmethod
    def batch_local_scores(self, target, parents, parent_nums):
        pass
    @abstractmethod
    def single_local_scores(self, target, parents,  parent_num):
        pass


def logdet(array):
    _, logdet = np.linalg.slogdet(array)
    return logdet
def ix_(array):
    return array[...,:,None],array[...,None,:]


class BGeScore(BaseScore):
    r"""BGe score.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the (continuous) dataset D. Each column
        corresponds to one node. The dataset D is assumed to only
        contain observational data_bio (a `INT` column will be treated as
        a continuous node like any other).

    prior : `BasePrior` instance
        The prior over graphs p(G).

    mean_obs : np.ndarray (optional)
        Mean parameter of the Normal prior over the mean $\mu$. This array must
        have size `(N,)`, where `N` is the number of nodes. By default,
        the mean parameter is 0.

    alpha_mu : float (default: 1.)
        Parameter $\alpha_{\mu}$ corresponding to the precision parameter
        of the Normal prior over the mean $\mu$.

    alpha_w : float (optional)
        Parameter $\alpha_{w}$ corresponding to the number of degrees of
        freedom of the Wishart prior of the precision matrix $W$. This
        parameter must satisfy `alpha_w > N - 1`, where `N` is the number
        of varaibles. By default, `alpha_w = N + 2`.
    """
    def __init__(self,data,prior,mean_obs=None,alpha_mu=1.,alpha_w=None):
        super().__init__(data, prior)
        num_nodes = len(data.columns)
        if mean_obs is None: mean_obs = np.zeros((num_nodes,))
        if alpha_w is None:  alpha_w = num_nodes + 2.

        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu
        self.alpha_w = alpha_w

        self.num_samples = self.data.shape[0]
        self.t = (self.alpha_mu * (self.alpha_w - self.num_nodes - 1)) / (self.alpha_mu + 1)
        #self.t=self.alpha_mu/(self.alpha_mu+self.num_samples)

        T = self.t * np.eye(self.num_nodes)
        #T = torch.eye(self.num_nodes)# assuem W^-1 of wishart prior is I
        data = np.asarray(self.data)
        data_mean = np.mean(data, axis=0, keepdims=True)
        data_centered = data - data_mean

        self.R = (T + np.dot(data_centered.T, data_centered)  #                        T+S_N
                  + ((self.num_samples * self.alpha_mu) / (self.num_samples + self.alpha_mu))
                  * np.dot((data_mean - self.mean_obs).T, data_mean - self.mean_obs)   # (N*α_μ)/(N+α_μ)*(v-x_μ)(v-x_μ)T
                  )
        self.block_R_I = block_diag(self.R, np.eye(self.num_nodes))
        all_parents = np.arange(self.num_nodes)
        self.log_gamma_term = (
            0.5 * (math.log(self.alpha_mu) - math.log(self.num_samples + self.alpha_mu))
            + gammaln(0.5 * (self.num_samples + self.alpha_w - self.num_nodes + all_parents + 1))
            - gammaln(0.5 * (self.alpha_w - self.num_nodes + all_parents + 1))
            - 0.5 * self.num_samples * math.log(math.pi)
            + 0.5 * (self.alpha_w - self.num_nodes + 2 * all_parents + 1) * math.log(self.t)
        )

    def batch_local_scores(self,target, parents, parent_nums):
        """
        batch-wise computing the local scores w.r.t. a node for a batch of graphs
        Idea: compute the logdet of [R   0],  the logdet (I) is 0. so, default parents indices are [n,...,2*n-1]
                                    [0,  I]
        when there are parents i in (0, n-1), change i_th elements of [n,...,i,...,2*n-1] to i.
        Since targets j must differ form their parents, we can also change j_th elements of [n,...,i,...,2*n-1] to j.

        Args:
            target  (int): The target node index.
            parents (array): The indices of the parents of the target node in each graph. (batch_size, num_nodes)
            parent_nums (array): The numbers of the parents of the target node in each graph. (batch_size, num_nodes)
        """
        batch_size  = len(parent_nums)
        num_parents = parent_nums.copy()
        nodes   = parents.copy()
        nodes[np.arange(batch_size),target]= target
        weight  = self.num_samples + self.alpha_w - self.num_nodes + num_parents
        log_term_r = (0.5 * (weight)* logdet(self.block_R_I[ix_(parents)])
                      - 0.5 * (weight + 1) * logdet(self.block_R_I[ix_(nodes)]))
        return self.log_gamma_term[num_parents] + log_term_r + self.prior(num_parents)

    def single_local_scores(self,target,parents,num_parent):
        """
        computing the local scores w.r.t. a node for a graph
        Args:
            target  (int): The target node index.
            parents (array): The indices of the parents of the target node in a graph. (num_nodes)
            parent_nums int: The number of the parents of the target node in a graph.
        """
        weight = self.num_samples + self.alpha_w - self.num_nodes
        if num_parent >0:
            nodes = np.concatenate((np.array([target]), parents))
            log_term_r = (0.5 * (weight+num_parent)  * logdet(self.R[ix_(parents)])    # log{ |R_YY|^(N+a_w-n+l)/2}
                          - 0.5 * (weight + num_parent + 1) * logdet(self.R[ix_(nodes)]))
        else:
            log_term_r = -0.5 * (weight+1) * np.log(np.abs(self.R[target, target]))
        return self.log_gamma_term[num_parent]  + log_term_r+ self.prior(num_parent)