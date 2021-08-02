import torch.nn as nn
from torch.distributions import Distribution
import torch


class Model(nn.Module):
    def compute_loss(self, data, data_idx):
        raise NotImplemented()


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

