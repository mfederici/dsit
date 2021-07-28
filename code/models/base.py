import torch.nn as nn
from torch.distributions import Distribution


class Model(nn.Module):
    def compute_loss(self, data, data_idx):
        raise NotImplemented()


class ConditionalDistribution(Model):
    def forward(self, *args, **kwargs) -> Distribution:
        pass


class MarginalDistribution(Model):
    def forward(self, *args, **kwargs) -> Distribution:
        pass


