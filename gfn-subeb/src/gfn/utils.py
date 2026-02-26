from typing import Dict, Optional,Tuple

import numpy as np
import torch,math
from src.gfn.containers.replay_buffer import Replay_x
from src.gfn.samplers import TrajectoriesSampler,DiscreteActionsSampler
from src.gfn.envs import Env,HyperGrid,DAG_BN,BioSeqEnv,BioSeqPendEnv
from src.gfn.losses import (
    Loss,
    Parametrization,
    FMParametrization,
    DBParametrization,
    TBParametrization,
    SubTBParametrization,
    RLParametrization
)
from src.gfn.distributions import Empirical_Dist,Empirical_Ratio


def JSD(P, Q):
    """Computes the Jensen-Shannon divergence between two distributions P and Q"""
    P=torch.maximum(P,torch.tensor(1e-20))
    Q=torch.maximum(Q,torch.tensor(1e-20))
    M = 0.5 * (P + Q)
    return 0.5 * (torch.sum(P * torch.log(P / M)) + torch.sum(Q * torch.log(Q / M)))
def split_to_chunk_list(states, min_len=int(1e6)):
    if len(states)>min_len:
        return [states[i:i + min_len] for i in range(0, len(states), min_len)]
    else:
        return [states]

def get_exact_P_T_Hypergrid(env,sampler):
    """This function evaluates the exact terminating state distribution P_T for HyperGrid.
    P_T(s') = u(s') P_F(s_f | s') where u(s') = \sum_{s \in Par(s')}  u(s) P_F(s' | s), and u(s_0) = 1
    """
    grid = env.build_grid()
    probabilities = sampler.actions_sampler.get_probs(grid)
    u = torch.ones(grid.batch_shape)
    all_states =env.all_states.states_tensor.tolist()
    for grid_ix in all_states[1:]:
        index = tuple(grid_ix)
        parents = [ index[:i] + (index[i] - 1,) + index[i + 1 :] + (i,)
                    for i in range(len(index)) if index[i] > 0] # parent, actions
        parents = torch.tensor(parents).T.numpy().tolist()
        u[index] = torch.sum(u[parents[:-1]] * probabilities[parents])
    return (u * probabilities[..., -1]).view(-1).detach().cpu()
def get_exact_P_T_bitseq(env,sampler):
    """
    This function evaluates the exact terminating state distribution P_T for graded DAG.
    :math:`P_T(s') = u(s') P_F(s_f | s')` where :math:`u(s') = \sum_{s \in Par(s')}  u(s) P_F(s' | s), and u(s_0) = 1`
    """
    ordered_states_list = env.ordered_states_list
    probabilities = sampler.actions_sampler.get_probs(env.all_states)
    u = torch.ones(size=(probabilities.shape[0],))
    for i, states in enumerate(ordered_states_list[1:]):
        #print(i + 1)
        index = env.get_states_indices(states)
        parents = torch.repeat_interleave(states.states_tensor,i+1,dim=0)

        backward_idx = torch.where(states.states_tensor != -1)[1]
        actions_idx =   env.ndim*parents[torch.arange(len(backward_idx)),backward_idx]+backward_idx
        parents[torch.arange(len(backward_idx)),backward_idx]=-1

        parents_idx = env.get_states_indices(env.States(parents))
        u[index] = (u[parents_idx] * probabilities[parents_idx, actions_idx]).reshape(-1, i + 1).sum(-1)

    return u[index].view(-1).cpu()

def get_exact_P_T_bitpend(env,sampler):
    """
    This function evaluates the exact terminating state distribution P_T for graded DAG.
    :math:`P_T(s') = u(s') P_F(s_f | s')` where :math:`u(s') = \sum_{s \in Par(s')}  u(s) P_F(s' | s), and u(s_0) = 1`
    """
    ordered_states_list = env.ordered_states_list
    u = torch.ones(size=(env.n_states,))
    # intial step
    for i, states in enumerate(ordered_states_list[1:]):
        parent_prob = torch.zeros(size=(env.n_ordered_states(i)-env.n_ordered_states(i-1), env.action_space.n))
        parent_prob[torch.tensor(env.get_states_indices(ordered_states_list[i]))-env.n_ordered_states(i-1)] \
            = sampler.actions_sampler.get_probs( ordered_states_list[i] )
        #####################################################################
        index = env.get_states_indices(states)
        parents = torch.repeat_interleave(states.states_tensor,2,dim=0)

        append_idx= (env.forward_index(states.states_tensor)-1).tolist()
        actions_idx  =  torch.stack([states.states_tensor[torch.arange(states.batch_shape[0]), append_idx]]
          + [states.states_tensor[:, 0] + env.nbase],dim=-1).flatten()

        odds_idx=torch.arange(1,states.batch_shape[0]*2,2)
        parents[ odds_idx,0:-1]= parents[ odds_idx,1:]
        parents[ odds_idx,-1]  =-1        #de-prepend
        parents[ odds_idx-1,append_idx] =-1 #de-append

        parents_idx = env.get_states_indices(env.States(parents))
        u[index] = (u[parents_idx] *  parent_prob[torch.tensor(parents_idx)-env.n_ordered_states(i-1), actions_idx]).reshape(-1, 2).sum(-1)
    return u[index].view(-1).cpu()

def get_exact_P_T(env, sampler):
    """This function evaluates the exact terminating state distribution P_T for DAG .
    P_T(s') = u(s') P_F(s_f | s') where u(s') = \sum_{s \in Par(s')}  u(s) P_F(s' | s), and u(s_0) = 1
    """
    all_states = env.all_states
    probabilities =  sampler.actions_sampler.get_probs(all_states)
    u = torch.ones(size=all_states.batch_shape)
    for i,state in enumerate(all_states[1:]):
        #print(i+1)
        parents = env.all_step(state[None,:], Backward=True)[state[None,:].backward_masks]
        parents_idx= env.get_states_indices(parents)
        actions_idx= torch.where(state[None,:].backward_masks)[1].tolist()
        u[i+1] = torch.sum(u[parents_idx] * probabilities[parents_idx,actions_idx])
    return (u * probabilities[..., -1]).view(-1).cpu()

def get_exact_P_T_G(env, sampler):
    """
    This function evaluates the exact terminating state distribution P_T for graded DAG.
    :math:`P_T(s') = u(s') P_F(s_f | s')` where :math:`u(s') = \sum_{s \in Par(s')}  u(s) P_F(s' | s), and u(s_0) = 1`
    """
    ordered_states_list = env.ordered_states_list
    probabilities =torch.zeros(size=(env.n_states,env.action_space.n))
    probabilities[env.get_states_indices(env.all_states)]=sampler.actions_sampler.get_probs(env.all_states)
    u=torch.ones(size=env.all_states.batch_shape)
    for i,states in enumerate(ordered_states_list[1:]):
        #print(i+1)
        num_parent_state=states.backward_masks.sum(-1)[0].item()# =i+1 for bitseq, =2 for bitpend
        index   =  env.get_states_indices(states)
        parents = env.all_step(states, Backward=True)[states.backward_masks]
        actions_idx= env.bction2action( env.States(torch.repeat_interleave(states.states_tensor,2,dim=0)),
                                        torch.where(states.backward_masks)[1])#.tolist()
        parents_idx= env.get_states_indices(parents)
        u[index] = (u[parents_idx] * probabilities[parents_idx,actions_idx]).reshape(-1, num_parent_state).sum(-1)

    return u[index].view(-1).cpu()

def validate_dist(
    env: Env,
    parametrization: Parametrization,
    sampler:TrajectoriesSampler,
    n_validation_samples: int = 1000,
    exact:bool=False,
    B_sampler:TrajectoriesSampler=None
) -> Tuple[Dict[str, float], float]:
    """Evaluates the current parametrization on the given environment.
    This is for environments with known target reward. The validation is done by computing the l1 distance between the
    learned empirical and the target distributions.

    Args:
        env: The environment to evaluate the parametrization on.
        parametrization: The parametrization to evaluate.
        n_validation_samples: The number of samples to use to evaluate the pmf.

    Returns:
        Dict[str, float]: A dictionary containing the l1 validation metric. If the parametrization is a TBParametrization,
        i.e. contains LogZ, then the (absolute) difference between the learned and the target LogZ is also returned in the
        dictionary.
    """
    validation_info= {}
    final_states_dist_pmf =None
    device=env.device
    env.to_device('cpu')
    if not exact:
        final_states = sampler.sample_T(n_trajectories=n_validation_samples)
        final_states_dist = Empirical_Ratio(env, sampler,B_sampler)
        final_states_dist_pmf = final_states_dist.pmf(final_states)
        true_dist_pmf = (env.log_reward(final_states) - env.log_reward(final_states).logsumexp(-1)).exp()
        validation_info["l1_dist"] = 0.5 * torch.abs(final_states_dist_pmf - true_dist_pmf).sum().item()
        if isinstance(env,BioSeqEnv) or isinstance(env,BioSeqPendEnv):
            final_states_dist= Empirical_Dist(env)
            final_states_dist_pmf = final_states_dist.pmf(final_states)
            validation_info["mean_diff"] = (env.log_reward(final_states).exp().mean() / env.mean_reward).clamp(0,1).item()
    else:
        true_dist_pmf = env.true_dist_pmf
        if isinstance(env, HyperGrid):
            final_states_dist_pmf = get_exact_P_T_Hypergrid(env, sampler)
        elif isinstance(env,DAG_BN):
            final_states_dist_pmf = get_exact_P_T(env, sampler)
        elif isinstance(env,BioSeqEnv):
            final_states_dist_pmf = get_exact_P_T_bitseq(env, sampler)
        elif isinstance(env,BioSeqPendEnv):
            final_states_dist_pmf = get_exact_P_T_bitpend(env, sampler)
        else:
            raise ValueError("Environment Not supported")
        if  isinstance(env,BioSeqEnv) or isinstance(env,BioSeqPendEnv):
            est_reward = (final_states_dist_pmf* env.log_reward(env.terminating_states).exp()).sum()
            validation_info["mean_diff"] =  (est_reward / env.mean_reward).clamp(0,1).item()
        validation_info["l1_dist"]= 0.5*torch.abs(final_states_dist_pmf - true_dist_pmf).sum().item()
        validation_info["JSD"]= JSD(final_states_dist_pmf,true_dist_pmf).item()
    if exact or isinstance(env,BioSeqEnv) or isinstance(env,BioSeqPendEnv):
        true_logZ = env.log_partition
        if isinstance(parametrization, TBParametrization) | isinstance(parametrization, RLParametrization):
            logZ = parametrization.logZ.tensor
        elif isinstance(parametrization,DBParametrization)|isinstance(parametrization,SubTBParametrization):
            logZ = parametrization.logF(env.States(env.s0))
        elif isinstance(parametrization,FMParametrization):
            logZ = parametrization.logF(env.States(env.s0)).logsumexp(-1)
        else:
            logZ = env.logZ_est
        validation_info["Z_diff"] = abs((logZ.exp() - true_logZ.exp()).item())
        validation_info["logZ_diff"] = abs((logZ - true_logZ).item())
    env.to_device(device)
    return validation_info, final_states_dist_pmf


#from polyleven import levenshtein
from itertools import permutations, product,chain,combinations

# def mean_pairwise_distances(seqs:Str):
#     dist=[]
#     for pair in combinations(seqs,2):
#         dist.append(levenshtein(*pair))
#     return np.mean(dist)

def mean_pairwise_distances(seqs):
    seqs = np.asarray(seqs)
    dist = []
    for a, b in combinations(seqs, 2):
        d = np.sum(a != b)
        dist.append(d)
    return np.mean(dist)

def validate_mode(
    env: Env,
    parametrization: Parametrization,
    sampler:TrajectoriesSampler,
    buffer: Replay_x,
) -> Dict[str, float]:
    validation_info = {}
    device=env.device
    env.to_device('cpu')
    if isinstance(env,BioSeqEnv) or isinstance(env,BioSeqPendEnv):
        validation_info["num_modes"]= env.oracle.is_mode(buffer.unique_states_rewards[0]).sum().item()#env.oracle.is_mode_r(buffer.x_rewards.unique()).sum().item()#
    else:
        replay_states,replay_rewards= buffer.unique_states_rewards
        sorted_idx=  replay_rewards.sort()[1][-100:]
        validation_info["mean_top_1"]   =   replay_rewards[sorted_idx[-1]].item()
        validation_info["mean_top_10"]  =   replay_rewards[sorted_idx[-10:]].mean().item()
        validation_info["mean_top_100"] =   replay_rewards[sorted_idx].mean().item()
        validation_info["diversity"]  = mean_pairwise_distances(replay_states[sorted_idx])

    if isinstance(parametrization, TBParametrization) | isinstance(parametrization, RLParametrization):
        logZ = parametrization.logZ.tensor
    elif isinstance(parametrization,DBParametrization)|isinstance(parametrization,SubTBParametrization):
        logZ = parametrization.logF(env.States(env.s0))
    else:
        logZ = env.logZ_est
    validation_info["logZ"] = logZ.item()
    env.to_device(device)
    return validation_info

import networkx as nx
def check_acylic(states_tensor):
    is_directed = []
    for edges in states_tensor:
        edges = edges.reshape( int(edges.shape[-1]**0.5),
                               int(edges.shape[-1]**0.5)).numpy()
        G = nx.DiGraph(edges)
        is_directed.append(nx.is_directed_acyclic_graph(G))
    return all(is_directed)

def all_dag(n_nodes,n_edges=25):
    nodelist = list(range(n_nodes))
    edges = list(permutations(nodelist, 2))  # n*(n-1) possible directed edges
    all_graphs = chain.from_iterable(combinations(edges, r) for r in range(len(edges) + 1)) #power set

    for graph_edges in all_graphs:
        if len(graph_edges)>n_edges:
            continue
        graph = nx.DiGraph(graph_edges)
        graph.add_nodes_from(nodelist)
        if nx.is_directed_acyclic_graph(graph):
            str_adj= nx.to_numpy_array(graph,dtype=int,nodelist=sorted(graph.nodes)).flatten()
            yield  str_adj

##################################
# for param operation
##################################
def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = param.numel()
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

