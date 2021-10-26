import numpy as np
import pandas as pd
import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig

from src.discrete.models.criteria import InformationBottleneckCriterion, IndependenceCriterion, \
                                          SufficiencyCriterion, SeparationCriterion
from src.discrete.models.training import train
from src.discrete.models.encoders import DiscreteEncoder
from src.discrete.distribution import compute_ce
from src.data.cmnist_dist import make_joint_distribution, CMNIST_VERSIONS
from tqdm.auto import tqdm


@hydra.main(config_path='discrete_config', config_name='config.yaml')
def parse(conf: DictConfig):
    criteria = [
        {
            'name': 'Information Bottleneck',
            'class': InformationBottleneckCriterion,
            'lambdas': np.exp(np.linspace(0, np.log(2), conf.n)) - 1
        },
        {
            'name': 'Independence',
            'class': IndependenceCriterion,
            'lambdas': np.exp(np.linspace(0, np.log(10 ** 6), conf.n)) - 1
        },
        {
            'name': 'Separation',
            'class': SeparationCriterion,
            'lambdas': np.exp(np.linspace(0, np.log(10 ** 6), conf.n)) - 1
        },
        {
            'name': 'Sufficiency',
            'class': SufficiencyCriterion,
            'lambdas': np.exp(np.linspace(0, np.log(10 ** 6), conf.n)) - 1
        }
    ]

    datasets = CMNIST_VERSIONS

    results = []

    for dataset_name in tqdm(datasets):
        tqdm.write(dataset_name)
        dist = make_joint_distribution(dataset_name)

        train_dist = dist.condition_on('t', 1).marginal(['x', 'y', 'e'])
        test_dist = dist.condition_on('t', 0).marginal(['x', 'y'])

        for criterion in tqdm(criteria):
            tqdm.write(criterion['name'])
            for l in tqdm(criterion['lambdas']):
                encoder = DiscreteEncoder(z_dim=conf.z_dim)

                train(encoder,
                      criterion['class'](l),  # use the specified criterion with regularization strenght "l"
                      train_dist=train_dist,  # train distribution
                      test_dist=test_dist,  # test distribution
                      verbose=conf.verbose,
                      lr=conf.lr,
                      tollerance=conf.tollerance,
                      initial_iterations=conf.initial_iterations,
                      step_iterations=conf.step_iterations)

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

    pd.DataFrame(results).to_csv(conf.savefile)


if __name__ == '__main__':
    parse()
