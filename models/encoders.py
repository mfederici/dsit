import torch
import torch.nn as nn
from utils.distribution import DiscreteDistribution

# Model for q(z|x) using a learnable weight matrix
class DiscreteEncoder(nn.Module):
    def __init__(self, z_dim=64):
        super(DiscreteEncoder, self).__init__()
        # define the encoding matrix
        q_z_x = torch.zeros(z_dim, 20)

        # initialize with random noise
        q_z_x.normal_()

        # wrap into an optimizable parameter matrix
        self.q_z_x = nn.Parameter(q_z_x)

    def forward(self, dist):
        # Normalize the encoding matrix
        q_z_x_normalized = self.q_z_x.softmax(0)

        # define the encoding distribution using the normalized matrix
        q_z_x = DiscreteDistribution(q_z_x_normalized, ['z', 'x'], condition=['x'])

        # compose the conditional distribution
        return dist.compose(q_z_x)
