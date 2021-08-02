import torch
from pyro.distributions import TransformModule
from torch.distributions import Distribution, TransformedDistribution
from torch.nn import ModuleList

from code.models.base import GenerativeModel
from torch import Tensor
from typing import List


class Flow(GenerativeModel):
    def __init__(
            self,
            base_distribution: Distribution,
            transforms: List[TransformModule]
    ):
        '''
        Generic flow model based on density transformation
        :param base_dist: the starting distribution to be transformed
        :param transform: the (invertible) transformation responsible for transforming base_dist
        '''
        super(Flow, self).__init__()

        self.transforms = ModuleList(transforms)
        self.dist = TransformedDistribution(base_distribution=base_distribution,
                                            transforms=list(self.transforms),
                                            validate_args=False)

    def forward(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def sample(self, sample_shape: torch.Size):
        return self.dist.sample(sample_shape=sample_shape)

    def encode(self, x) -> Tensor:
        return self(x)

    def decode(self, z):
        for transform in reversed(self.transforms):
            z = transform.inv(z)
        return z

    def compute_loss(self, data, data_idx):
        x = data['x']
        #x = x.view(x.shape[0], -1)
        log_p_x = self.dist.log_prob(x)
        return {'loss': -log_p_x.mean()}

    def __repr__(self):
        return 'Flow (%s)' % self.transforms.__repr__()
