from typing import ClassVar, Literal, Tuple, Union,Optional
import torch
from gymnasium.spaces import Discrete
from torchtyping import TensorType

from src.gfn.containers.states import States
from src.gfn.envs.env import Env
from src.gfn.envs.preprocessors import IdentityPreprocessor
import numpy as np
from functorch import vmap
# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]

from math import comb
def count_dags(n: int) -> int:
    if n == 0: return 1
    total = 0
    for k in range(1, n + 1):
        total += ((-1) ** (k + 1)) * comb(n, k) * (2 ** (k * (n - k))) * count_dags(n - k)
    return total

def nbase2dec(n: int, b: torch.Tensor, length: int):
    """
    Convert base-n digits to decimal.
    b: shape (..., length), digits in [0, n-1]
    returns: shape (...) int64
    """
    canonical_base = n ** torch.arange(length).to(b.device, b.dtype)
    return torch.sum(canonical_base * b, -1)

def dec2nbase(code: torch.Tensor, n: int, length: int) -> torch.Tensor:
    """
    Convert decimal to base-n with fixed length.
    code: shape (...), int64
    returns: shape (..., length), digits in [0, n-1]
    """
    code = code.to(torch.int64)
    digs = []
    for _ in range(length):
        digs.append(code % n)
        code //= n
    return torch.stack(digs[::-1], dim=-1)

class DAG_BN(Env):
    def __init__(
            self,
            ndim,
            all_graphs,
            true_graph,
            max_parents=None,
            max_edges =None,
            device_str: Literal["cpu", "cuda"] = "cpu",
            score=None,
            alpha=1.,
            reward_types='vanila'
    ):
        """GFlowNet environment for learning a distribution over DAGs.

        Parameters
        ----------
        scorer : BaseScore instance
            The score to use. Note that this contains the data_bio.

        ndim :  int, optional
            num_of variables.

        max_parents : int, optional
            Maximum number of parents for each node in the DAG. If None, then
            there is no constraint on the maximum number of parents.
        States: flattened adjency matrix of   ⬇ source nodes  -> target nodes
        """
        self.reward_types= reward_types
        self.alpha=alpha
        self.ndim=ndim
        self.max_parents = self.ndim if max_parents is None else max_parents
        self.max_edges   = self.ndim * (self.ndim - 1) // 2  if max_edges is None  else max_edges # dag edge limite
        self.zero_score  =score(torch.zeros(self.ndim**2)).item()
        self.max_score   =score(true_graph.flatten()).item()
        self.score       =score
        self.true_graph  =true_graph
        self.all_graphs  =all_graphs
        self.all_indices = {idx: val for val, idx in
                           enumerate(nbase2dec(2,all_graphs,all_graphs[0].shape[0]).tolist())}
        #self.all_indices ={np.array2string(item,separator=','): idx for idx, item in enumerate(all_graphs.numpy())}
        preprocessor = IdentityPreprocessor(output_shape=(ndim**2,))
        action_space = Discrete(self.ndim ** 2 + 1)                   # all possible edges+stop action

        s0 = torch.zeros((ndim*ndim,), dtype=torch.long, device=torch.device(device_str))
        sf = torch.full( (ndim*ndim,), fill_value=-1, dtype=torch.long, device=torch.device(device_str))

        super().__init__(
            action_space=action_space,
            s0=s0,
            sf=sf,
            device_str=device_str,
            preprocessor=preprocessor,
        )

    def make_States_class(self) -> type[States]:
        "Creates a States class for this environment"
        env = self
        class DAG_States(States):

            state_shape: ClassVar[tuple[int, ...]] = (env.ndim*env.ndim,)
            s0:ClassVar[OneStateTensor]            = env.s0
            sf:ClassVar[OneStateTensor]            = env.sf

            def __init__(
                    self,
                    states_tensor: StatesTensor,
                    forward_masks: ForwardMasksTensor | None = None,
                    backward_masks: BackwardMasksTensor | None = None,
                    forward_closure_T: ForwardMasksTensor | None = None,
            ):
                super().__init__(states_tensor,
                                 forward_masks,
                                 backward_masks)
                # forward_closure is the new_closure of current states after updating mask,
                # and is the old_closure of previous states before updating mask, and after maskless_step
                if forward_closure_T is None:
                    self.forward_closure_T= (1-self.forward_masks[...,:-1].int()-self.states_tensor).bool()
                else:
                    self.forward_closure_T= forward_closure_T

            @classmethod
            def make_random_states_tensor(cls, batch_shape: Tuple[int, ...]) -> StatesTensor:
                "Creates a batch of random states."
                states_tensor = torch.randint(0, 2, batch_shape + env.s0.shape, device=env.device)
                return states_tensor

            def closure_T_exact(self,adjacency):
                '''
                This computes the new_closure based on only adjacency matrix. This is computationally expensive
                Therefore, in the function update_masks, we compute new_closure by recorded old_closure and current actions
                '''
                reach =adjacency.reshape(*adjacency.shape[0:-1], env.ndim, env.ndim).transpose(-1,-2).bool()
                reach = reach | torch.eye(env.ndim, dtype=torch.bool)

                for k in range(env.ndim):
                    reach_k  = reach[...,:,[k]]
                    k_reach  = reach[...,[k],:]
                    reach    = reach | (reach_k & k_reach)
                return reach.reshape(*adjacency.shape[0:-1],env.ndim**2)

            def make_masks(self) -> Tuple[ForwardMasksTensor, BackwardMasksTensor]:
                "Mask illegal (forward and backward) actions."
                backward_masks=self.states_tensor.bool()
                forward_masks = torch.ones((*self.batch_shape, env.n_actions), dtype=torch.bool, device=env.device)
                new_masks=1-(self.states_tensor+self.closure_T_exact(self.states_tensor))
                forward_masks[...,:-1]=new_masks.bool()
                forward_masks[...,-1] =self.states_tensor.sum(-1)>1
                return forward_masks,backward_masks

            def update_masks(self,actions:TensorLong,index=TensorBool):
                """
                Update the masks based on the current states. Here we don't distinguish sink_states and other states
                as the for/backward_mask and forward_closure of sink_states does not matter.
                """
                sources, targets = torch.div(actions[index], env.ndim, rounding_mode='floor'), \
                                   torch.fmod(actions[index], env.ndim)
                adjcency=self.states_tensor.reshape(*self.batch_shape, env.ndim, env.ndim)[index]
                #
                old_closure_T=self.forward_closure_T.reshape(*self.batch_shape, env.ndim, env.ndim)[index]
                source_rows = old_closure_T[torch.arange(index.sum()), sources, :][...,None,:]  # insert one dim  [num_env,1,num_variables]
                target_cols = old_closure_T[torch.arange(index.sum()), :, targets][...,:,None]
                new_closure_T = torch.logical_or(old_closure_T, source_rows.logical_and(target_cols))
                # Update the masks (maximum number of parents)
                num_parents = torch.sum(adjcency, dim=-2, keepdim=True)
                num_edges   = torch.sum(adjcency, dim=-2,keepdim=True).sum(dim=-1,keepdim=True)
                # Update the masks
                new_masks   =1- (new_closure_T+adjcency)
                # exact_masks = 1-(self.states_tensor[index]+ self.closure_T_exact(self.states_tensor[index]))
                # assert ( exact_masks == new_masks.flatten(-2) ).all()
                new_masks   =new_masks.mul(num_parents < env.max_parents)
                new_masks   =new_masks.mul(num_edges  < env.max_edges)
                self.forward_masks[...,:-1][index]= new_masks.reshape(-1,env.ndim**2).bool()
                self.forward_masks[...,-1] =self.states_tensor.sum(-1)>1
                self.forward_closure_T[index]=new_closure_T.reshape(-1, env.ndim**2)
                self.backward_masks = self.states_tensor.bool()

            def backward_update_masks(self,bctions:TensorLong,index=TensorBool):
                self.forward_closure_T[index] =  self.closure_T_exact(self.states_tensor[index])
                new_masks = 1 - (self.states_tensor[index] +  self.forward_closure_T[index] )
                self.forward_masks[...,:-1][index]= new_masks.bool()
                self.backward_masks = self.states_tensor.bool()
        return DAG_States

    # def all_states(self,n_edges=25):
    #     nodelist = list(range(self.n_dim))
    #     edges = list(permutations(nodelist, 2))  # n*(n-1) possible directed edges
    #     all_graphs = chain.from_iterable(combinations(edges, r) for r in range(len(edges) + 1)) #power set
    #
    #     for graph_edges in all_graphs:
    #         if len(graph_edges)>n_edges:
    #             break
    #         graph = nx.DiGraph(graph_edges)
    #         graph.add_nodes_from(nodelist)
    #         if nx.is_directed_acyclic_graph(graph):
    #             str_adj= nx.to_numpy_array(graph,dtype=int,nodelist=sorted(graph.nodes)).flatten()
    #             yield  np.array2string(str_adj,separator=','),torch.from_numpy(str_adj)

    def maskless_step(self, states: StatesTensor,actions:TensorLong) -> StatesTensor:
        index = (torch.arange(0,  actions.shape[0]))
        states[...,index,actions]= 1
        return states

    def maskless_backward_step(self, states: StatesTensor,actions:TensorLong) -> StatesTensor:
        index = (torch.arange(0,  actions.shape[0]))
        states[...,index,actions]= 0
        return states

    def get_states_indices(self, states: States):
        indices    =  nbase2dec(2, states.states_tensor,self.ndim*self.ndim).long().cpu().tolist()
        indices    =  [self.all_indices[idx] for idx in indices]
        return indices

    # def get_states_indices(self, states: States):
    #     indices    = [self.all_indices.get(np.array2string(i,separator=',')) for i in states.states_tensor.numpy()]
    #     return indices

    def get_terminating_states_indices(self, states: States):
        return self.get_states_indices(states)

    @property
    def all_states(self):
        return self.States(self.all_graphs)
    @property
    def terminating_states(self) -> States:
        return self.States(self.all_graphs)

    @property
    def n_terminating_states(self) -> int:
        return count_dags(self.ndim)

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        log_reward=self.log_reward(self.terminating_states)
        true_dist = log_reward-torch.logsumexp(log_reward,-1)
        return true_dist.exp().cpu()#,idx_list

    def log_reward(self, final_states:States):
        if  self.reward_types=='vanila':
            reward = self.score(final_states.states_tensor)-self.max_score
            return  reward/self.alpha
        else:
            reward = (self.score(final_states.states_tensor) - self.zero_score).clamp(min=0.)
            return ((reward / (self.max_score - self.zero_score)).log() * self.alpha).clamp(min=-100)

    @property
    def log_partition(self) -> torch.float:
        log_rewards = self.log_reward(self.terminating_states)
        log_Z = torch.logsumexp(log_rewards, -1)
        assert log_Z < 89., 'true logZ is too large, making Z to numerical infinity'
        return log_Z
    @property
    def mean_log_reward(self)->torch.float:
        return (self.true_dist_pmf * self.log_reward(self.terminating_states)).sum()