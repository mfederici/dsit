import torch.nn as nn
from examples.architectures.utils import Flatten, StochasticLinear, Reshape, make_stack
from torch.distributions import Normal, Independent
from examples.models.types import ConditionalDistribution


INPUT_SHAPE = [1, 28, 28]
N_INPUTS = 28*28
N_LABELS = 10


# Model for q(Z|X)
class Encoder(ConditionalDistribution):
    def __init__(self, z_dim: int, layers: list):
        '''
        Encoder network used to parametrize a conditional distribution
        :param z_dim: number of dimensions for the latent distribution
        :param layers: list describing the layers
        '''
        super(Encoder, self).__init__()

        # Create a stack of layers with ReLU activations as specified
        nn_layers = make_stack([N_INPUTS] + list(layers))

        self.net = nn.Sequential(
            Flatten(),  # Layer to flatten the input
            *nn_layers,  # The previously created stack
            nn.ReLU(True),  # A ReLU activation
            StochasticLinear(layers[-1], z_dim, 'Normal')  # A layer that returns a factorized Normal distribution
        )

    def forward(self, x):
        # Note that the encoder returns a factorized normal distribution and not a vector
        return self.net(x)


# Model for p(X|Z)
class Decoder(ConditionalDistribution):
    def __init__(self, z_dim: int, layers: list, sigma: int=1):
        super(Decoder, self).__init__()

        # Create a stack of layers with ReLU activations as specified
        nn_layers = make_stack([z_dim] + list(layers) + [N_INPUTS])

        self.net = nn.Sequential(
            *nn_layers,                 # The previously created stack
            Reshape([N_INPUTS], INPUT_SHAPE)        # A layer to reshape to the correct image shape
        )
        self.sigma = sigma

    def forward(self, x):
        # Note that the decoder returns a factorized normal distribution and not a vector
        # the last 3 dimensions (n_channels x x_dim x y_dim) are considered to be independent
        return Independent(Normal(self.net(x), self.sigma), 3)


# q(Y|Z)
class LatentClassifier(ConditionalDistribution):
    def __init__(self, z_dim: int, layers: list):
        super(LatentClassifier, self).__init__()

        # Create a stack of layers with ReLU activations as specified
        nn_layers = make_stack([z_dim] + list(layers))

        self.net = nn.Sequential(
            *nn_layers,  # The previously created stack
            nn.ReLU(True),  # A ReLU activation
            StochasticLinear(layers[-1], N_LABELS, 'Categorical')  # A layer that returns a Categorical distribution
        )

    def forward(self, z):
        # Note that the encoder returns a Categorical distribution and not a vector
        return self.net(z)


# Model for q(Y|X)
class Classifier(ConditionalDistribution):
    def __init__(self, layers: list):
        '''
        Encoder network used to parametrize a conditional distribution
        :param layers: list describing the layers
        '''
        super(Classifier, self).__init__()

        # Create a stack of layers with ReLU activations as specified
        nn_layers = make_stack([N_INPUTS] + list(layers))

        self.net = nn.Sequential(
            Flatten(),  # Layer to flatten the input
            *nn_layers,  # The previously created stack
            nn.ReLU(True),  # A ReLU activation
            StochasticLinear(layers[-1], N_LABELS, 'Categorical')  # A layer that returns a Categorical on 10 classes
        )

    def forward(self, x):
        # Note that the encoder returns a factorized normal distribution and not a vector
        return self.net(x)

