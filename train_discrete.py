import numpy as np
import pandas as pd

from src.discrete.models.criteria import InformationBottleneckCriterion, IndependenceCriterion, \
                                          SufficiencyCriterion, SeparationCriterion
from src.discrete.models.training import train
from src.discrete.models.encoders import DiscreteEncoder
from src.discrete.distribution import compute_ce
from src.data.cmnist_dist import make_joint_distribution, CMNIST_VERSIONS
from tqdm.auto import tqdm

n = 50
criteria = [
    {
        'name': 'Information Bottleneck',
        'class': InformationBottleneckCriterion,
        'lambdas': np.exp(np.linspace(0, np.log(2), n))-1
    },
    {
        'name': 'Independence',
        'class': IndependenceCriterion,
        'lambdas': np.exp(np.linspace(0, np.log(10**6), n))-1
    },
    {
        'name': 'Separation',
        'class': SeparationCriterion,
        'lambdas': np.exp(np.linspace(0, np.log(10**6), n))-1
    },
    {
        'name': 'Sufficiency',
        'class': SufficiencyCriterion,
        'lambdas': np.exp(np.linspace(0, np.log(10**6), n))-1
    }
]

z_dim = 64

results = []

for dataset_name in tqdm(CMNIST_VERSIONS):
    tqdm.write(dataset_name)
    dist = make_joint_distribution(dataset_name)

    train_dist = dist.condition_on('t', 1).marginal(['x', 'y', 'e'])
    test_dist = dist.condition_on('t', 0).marginal(['x', 'y'])

    for criterion in tqdm(criteria):
        tqdm.write(criterion['name'])
        for l in tqdm(criterion['lambdas']):
            encoder = DiscreteEncoder(z_dim=z_dim)

            logs = train(encoder,
                         criterion['class'](l),  # use the specified criterion with regularization strenght "l"
                         train_dist=train_dist,  # train distribution
                         test_dist=test_dist,  # test distribution
                         verbose=False)

            cond_dist = encoder(train_dist).conditional('y', 'z')

            ce_t1 = compute_ce(encoder(train_dist), cond_dist).item()
            ce_t0 = compute_ce(encoder(test_dist), cond_dist).item()

            results.append({
                'criterion': criterion['name'],
                'lambda': l,
                'dataset': dataset_name,
                'CrossEntropy(t=1)': ce_t1,
                'CrossEntropy(t=0)': ce_t0,
                'p': encoder.q_z_x.data
            })

pd.DataFrame(results).to_csv('results/discrete.csv')