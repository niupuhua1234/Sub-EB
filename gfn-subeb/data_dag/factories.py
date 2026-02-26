import pandas as pd
import urllib.request
import gzip

def download(url, filename):
    if filename.is_file():
        return filename
    filename.parent.mkdir(exist_ok=True)

    # Download & uncompress archive
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read()
    with open(filename, 'wb') as f:
        f.write(file_content)

    return filename

from pathlib import Path
from numpy.random import default_rng
from pgmpy.utils import get_example_model
from data_dag.graph import sample_erdos_renyi_linear_gaussian, sample_from_linear_gaussian

def get_data(name, args, rng=default_rng()):
    if name == 'erdos_renyi_lingauss':
        graph = sample_erdos_renyi_linear_gaussian(
            num_variables=args.num_variables,
            num_edges=args.num_edges,
            loc_edges=0.0,
            scale_edges=1.0,
            obs_noise=0.1,
            rng=rng
        )
        data = sample_from_linear_gaussian(graph, num_samples=args.num_samples, rng=rng)
        score = 'bge'

    elif name == 'sachs_continuous':
        graph = get_example_model('sachs')
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz',
            Path('data_dag/sachs.data_bio.txt')
        )
        data = pd.read_csv(filename, delimiter='\t', dtype=float)
        data = (data - data.mean()) / data.std()  # Standardize data_bio
        score = 'bge'

    elif name == 'sachs_discrete':
        graph = get_example_model('sachs')
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz',
            Path('data_dag/sachs.interventional.txt')
        )
        data = pd.read_csv(filename, delimiter=' ', dtype='category').iloc[:, 4:9]
        score = 'bde'

    else:
        raise ValueError(f'Unknown graph type: {name}')

    return graph, data, score


from numpy.random import default_rng
from data_dag.scores import BGeScore,priors

def get_prior(name, **kwargs):
    prior = {
        'uniform': priors.UniformPrior,
        'erdos_renyi': priors.ErdosRenyiPrior,
        'edge': priors.EdgePrior,
        'fair': priors.FairPrior
    }
    return prior[name](**kwargs)

def get_scorer(args,rng=default_rng(0)):
    # Get the data_bio
    graph, data, score = get_data(args.graph, args, rng=rng)
    data = data[list(graph.nodes)]
    # Get the prior
    prior  = get_prior(args.prior, **args.prior_kwargs)
    scorer = BGeScore(data, prior, **args.scorer_kwargs)

    return scorer, data, graph