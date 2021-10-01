import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from src.models import MarginalDistribution


# Model for p(Z)
class DiagonalNormal(MarginalDistribution):
    def __init__(self, z_dim: int):
        super(DiagonalNormal, self).__init__()

        self.mu = nn.Parameter(torch.zeros([1, z_dim]), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones([1, z_dim]), requires_grad=False)

    def forward(self):
        # Return a factorized Normal prior
        return Independent(Normal(self.mu, self.sigma), 1)