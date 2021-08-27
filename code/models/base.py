import torch.nn as nn
from torch.distributions import Distribution


class Model(nn.Module):
    def compute_loss(self, data, data_idx):
        raise NotImplemented()

class RegularizedModel(Model):
    def __init__(self, beta):
        super(RegularizedModel, self).__init__()
        self.beta = beta

    def compute_loss(self, data, data_idx):
        loss_components = self.compute_loss_components(data)

        loss = loss_components['reconstruction'] + self.beta * loss_components['regularization']

        return {
            'loss': loss,
            'reconstruction': loss_components['reconstruction'].item(),
            'regularization': loss_components['regularization'].item()
        }

    def compute_loss_components(self, data):
        raise NotImplemented()


class ConditionalDistribution(Model):
    def forward(self, *args, **kwargs) -> Distribution:
        pass


class MarginalDistribution(Model):
    def forward(self, *args, **kwargs) -> Distribution:
        pass

