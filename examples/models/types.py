from torch.distributions import Distribution
import torch

from framework.model.base import Model


class GenerativeModel(Model):
    def sample(self, shape: torch.Size):
        raise NotImplemented()


class RepresentationLearningModel(Model):
    def encode(self, x: torch.Tensor):
        raise NotImplemented


class ConditionalDistribution(Model):
    def forward(self, *args, **kwargs) -> Distribution:
        pass


class MarginalDistribution(Model):
    def forward(self, *args, **kwargs) -> Distribution:
        pass

