import torch
import torch.nn as nn
from code.architectures.utils import Flatten, StochasticLinear, make_stack, OneHot
from code.models.base import ConditionalDistribution

INPUT_SHAPE = [2, 28, 28]
N_INPUTS = 2*28*28
N_LABELS = 2
N_TRAIN_ENVS = 2


# Model for q(Z|X)
class Encoder(ConditionalDistribution):
    def __init__(self, z_dim: int, layers: list, dropout: float = 0.0, posterior: str = 'Normal'):
        '''
        Encoder network used to parametrize a conditional distribution
        :param z_dim: number of dimensions for the latent distribution
        :param layers: list describing the layers
        '''
        super(Encoder, self).__init__()

        # Create a stack of layers with ReLU activations as specified
        nn_layers = make_stack([N_INPUTS] + list(layers), dropout=dropout)

        self.net = nn.Sequential(
            Flatten(),  # Layer to flatten the input
            *nn_layers,  # The previously created stack
            nn.Dropout(dropout),
            nn.ReLU(True),  # A ReLU activation
            StochasticLinear(layers[-1], z_dim, posterior)  # A layer that returns a parametrized distribution
        )

    def forward(self, x):
        # Note that the encoder returns a factorized normal distribution and not a vector
        return self.net(x)


# q(Y|Z)
class LatentClassifier(ConditionalDistribution):
    def __init__(self, z_dim: int, layers: list, dropout: float = 0.0):
        super(LatentClassifier, self).__init__()

        # Create a stack of layers with ReLU activations as specified
        nn_layers = make_stack([z_dim] + list(layers), dropout=dropout)

        self.net = nn.Sequential(
            *nn_layers,  # The previously created stack
            nn.ReLU(True),  # A ReLU activation
            StochasticLinear(layers[-1], N_LABELS, 'Categorical')  # A layer that returns a Categorical distribution
        )

    def forward(self, z):
        # Note that the encoder returns a Categorical distribution and not a vector
        return self.net(z)


# Model for q(E|Z)
class Discriminator(ConditionalDistribution):
    def __init__(self, z_dim: int, layers: list, dropout: float = 0.0, spectral_norm=True):
        super(Discriminator, self).__init__()

        # Create a stack of layers with ReLU activations as specified
        nn_layers = make_stack([z_dim] + list(layers), dropout=dropout, spectral_norm=spectral_norm)

        self.net = nn.Sequential(
            *nn_layers,  # The previously created stack
            nn.ReLU(True),  # A ReLU activation
            StochasticLinear(layers[-1], N_TRAIN_ENVS, 'Categorical')  # A layer that returns a Categorical distribution
        )

    def forward(self, z):
        # Note that the encoder returns a Categorical distribution and not a vector
        return self.net(z)


# Model for q(E|YZ)
class JointDiscriminator(Discriminator):
    def __init__(self, z_dim: int, **kwargs):
        super(JointDiscriminator, self).__init__(z_dim=z_dim+N_TRAIN_ENVS, **kwargs)
        self.one_hot = OneHot(N_LABELS)

    def forward(self, z, y):
        y = self.one_hot(y.long())
        zy = torch.cat([z, y], 1)
        # Note that the encoder returns a Categorical distribution and not a vector
        return self.net(zy)
