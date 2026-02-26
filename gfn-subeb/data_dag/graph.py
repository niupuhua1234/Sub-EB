import string
import numpy as np
import pandas as pd
import networkx as nx
from numpy.random import default_rng
from pgmpy import models
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork, BayesianNetwork
from pgmpy.sampling import BayesianModelSampling

from itertools import chain, product, islice, count

def sample_erdos_renyi_graph(
        num_variables,
        p=None,
        num_edges=None,
        nodes=None,
        create_using=models.BayesianNetwork,
        rng=default_rng()
):
    """
    Sample an Erdos-Renyi graph.
    Args:
        num_nodes: Number of nodes in the graph.
        rng: Numpy random number generator.
        p: Probability of creating an edge.
        num_edges: Total number of edges (used to compute p if p is None).
        node_names: Optional list of node names.
    """
    if p is None:
        if num_edges is None:
            raise ValueError('One of p or num_edges must be specified.')
        p = num_edges / ((num_variables * (num_variables - 1)) / 2.)

    if nodes is None:
        uppercase = string.ascii_uppercase
        iterator = chain.from_iterable(
            product(uppercase, repeat=r) for r in count(1))
        nodes = [''.join(letters) for letters in islice(iterator, num_variables)]

    adjacency = rng.binomial(1, p=p, size=(num_variables, num_variables))
    adjacency = np.tril(adjacency, k=-1)  # Only keep the lower triangular part

    # Permute the rows and columns
    perm = rng.permutation(num_variables)
    adjacency = adjacency[perm, :]
    adjacency = adjacency[:, perm]

    graph = nx.from_numpy_array(adjacency, create_using=create_using)
    mapping = dict(enumerate(nodes))
    nx.relabel_nodes(graph, mapping=mapping, copy=False)

    return graph


def sample_erdos_renyi_linear_gaussian(
        num_variables,
        p=None,
        num_edges=None,
        nodes=None,
        loc_edges=0.0,
        scale_edges=1.0,
        obs_noise=0.1,
        rng=default_rng()
):
    """
    Sample a linear Gaussian Bayesian network based on an Erdos-Renyi graph.

    Args:
        num_nodes: Number of nodes.
        rng: Random number generator for reproducibility.
        p: Probability of creating an edge.
        num_edges: Total number of edges (used to compute p if p is None).
        node_names: Optional list of node names.
        loc_edges: Mean value for edge parameters.
        scale_edges: Standard deviation for edge parameters.
        obs_noise: Observation noise for each node.
    """
    # Create graph structure
    graph = sample_erdos_renyi_graph(
        num_variables,
        p=p,
        num_edges=num_edges,
        nodes=nodes,
        create_using=models.LinearGaussianBayesianNetwork,
        rng=rng
    )

    # Create the model parameters
    factors = []
    for node in graph.nodes:
        parents = list(graph.predecessors(node))

        # Sample random parameters (from Normal distribution)
        theta = rng.normal(loc_edges, scale_edges, size=(len(parents) + 1,))
        theta[0] = 0.  # There is no bias term

        # Create factor
        factor = LinearGaussianCPD(node, theta, obs_noise, parents)
        factors.append(factor)

    graph.add_cpds(*factors)
    return graph

def sample_from_linear_gaussian(model, num_samples, rng=default_rng()):
    """Sample from a linear-Gaussian model using ancestral sampling."""
    if not isinstance(model, LinearGaussianBayesianNetwork):
        raise ValueError('The model must be an instance '
                         'of LinearGaussianBayesianNetwork')

    samples = pd.DataFrame(columns=list(model.nodes()))
    for node in nx.topological_sort(model):
        cpd = model.get_cpds(node)
        if cpd.evidence:
            values = np.vstack([samples[parent] for parent in cpd.evidence])
            mean = cpd.mean[0] + np.dot(cpd.mean[1:], values)
            samples[node] = rng.normal(mean, cpd.variance)
        else:
            samples[node] = rng.normal(cpd.mean[0], cpd.variance, size=(num_samples,))
    return samples


def sample_from_discrete(model, num_samples, rng=default_rng(), **kwargs):
    """Sample from a discrete model using ancestral sampling."""
    if not isinstance(model, BayesianNetwork):
        raise ValueError('The model must be an instance of BayesianNetwork')
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=num_samples, show_progress=False, **kwargs)
    # Convert values to pd.Categorical for faster operations
    for node in samples.columns:
        cpd = model.get_cpds(node)
        samples[node] = pd.Categorical(samples[node], categories=cpd.state_names[node])
    return samples