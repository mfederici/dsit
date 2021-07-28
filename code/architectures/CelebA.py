import torch
import torch.nn as nn
from code.architectures.utils import Flatten, StochasticLinear, Reshape, make_stack, make_cnn_stack, make_cnn_deconv_stack
from torch.distributions import Normal, Independent

from code.models import ConditionalDistribution

INPUT_SHAPE = [3, 218, 218]
N_LANDMARKS = 5


# Model for q(Z|X)
class Encoder(ConditionalDistribution):
    def __init__(self, z_dim, layers):
        super(Encoder, self).__init__()

        # Create a stack of layers with ReLU activations as specified
        cnn_layers = make_cnn_stack(layers)

        self.net = nn.Sequential(
            *cnn_layers,  # The previously created stack
            Flatten(),  # Layer to flatten the input
            StochasticLinear(layers[-1]['out_channels'], z_dim, 'Normal')  # A layer that returns a factorized Normal distribution
        )

    def forward(self, x):
        # Note that the encoder returns a factorized normal distribution and not a vector
        return self.net(x)


# Model for p(X|Z)
class Decoder(ConditionalDistribution):
    def __init__(self, z_dim, layers, sigma):
        super(Decoder, self).__init__()

        # Create a stack of layers with ReLU activations as specified
        cnn_layers = make_cnn_deconv_stack(layers)

        self.net = nn.Sequential(
            Reshape([z_dim, 1, 1]),
            nn.Conv2d(in_channels=z_dim, out_channels=layers[0]['in_channels'], kernel_size=1),
            * cnn_layers                 # The previously created stack
        )
        self.sigma = sigma

    def forward(self, x):
        return Independent(Normal(self.net(x), self.sigma), 3)


class LandmarkPredictor(ConditionalDistribution):
    def __init__(self, z_dim, layers):
        super(LandmarkPredictor, self).__init__()

        # Create a stack of layers with ReLU activations as specified
        nn_layers = make_stack([z_dim] + list(layers))

        self.net = nn.Sequential(
            *nn_layers,  # The previously created stack
            nn.ReLU(True),  # A ReLU activation
            StochasticLinear(layers[-1], N_LANDMARKS*2, 'Normal')  # A layer that returns a Normal distribution
        )

    def forward(self, x):
        # Note that the encoder returns a Normal distribution and not a vector
        return self.net(x)

