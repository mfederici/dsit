import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, Beta, Categorical, Bernoulli
from pyro.distributions import Delta
from torch.nn.functional import softplus


# Create simple layer stacks with relu activations
def make_stack(layers):
    nn_layers = []
    for i in range(len(layers)-1):
        nn_layers.append(nn.Linear(layers[i], layers[i+1]))
        if i < len(layers)-2:
            nn_layers.append(nn.ReLU(True))

    return nn_layers


def make_cnn_stack(layers):
    cnn_layers = []
    for i in range(len(layers)):
        cnn_layers.append(nn.Conv2d(**layers[i]))
        if i < len(layers)-2:
            cnn_layers.append(nn.ReLU(True))

    return cnn_layers


def make_cnn_deconv_stack(layers):
    cnn_layers = []
    for i in range(len(layers)):
        cnn_layers.append(nn.ConvTranspose2d(**layers[i]))
        if i < len(layers)-2:
            cnn_layers.append(nn.ReLU(True))

    return cnn_layers


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(-1, *self.shape)


class Permute(nn.Module):
    def __init__(self, *permutation):
        super(Permute, self).__init__()
        self.permutation = permutation

    def forward(self, input):
        return input.permute(self.permutation)


class OneHot(nn.Module):
    def __init__(self, n_values):
        super(OneHot, self).__init__()
        self.eye = nn.Parameter(torch.eye(n_values), requires_grad=False)

    def forward(self, input):
        assert isinstance(input, torch.LongTensor) or isinstance(input, torch.cuda.LongTensor)
        return self.eye[input]


class StopGrad(nn.Module):
    def forward(self, x):
        return x.detach()


class StochasticLinear(nn.Linear):
    def __init__(self, in_size, out_size, dist):
        self.out_size = out_size
        self.dist = dist

        if dist == 'Normal' or dist == 'Beta':
            self.n_params = 2
        elif dist == 'Delta' or dist == 'Categorical' or dist == 'Bernoulli':
            self.n_params = 1
        else:
            raise NotImplementedError('"%s"' % dist)

        super(StochasticLinear, self).__init__(in_size, out_size*self.n_params)

    def forward(self, input):
        params = super(StochasticLinear, self).forward(input)
        params = torch.split(params, [self.out_size] * self.n_params, 1)

        if self.dist == 'Normal':
            mu, sigma = params[0], softplus(params[1]) + 1e-7
            dist = Normal(loc=mu, scale=sigma)
            dist = Independent(dist, 1)  # Factorized Normal distribution
        elif self.dist == 'Beta':
            c1, c0 = softplus(params[0]) + 1e-7, softplus(params[1]) + 1e-7
            dist = Beta(c1, c0)
            dist = Independent(dist, 1)  # Factorized Beta distribution
        elif self.dist == 'Delta':
            m = params[0]
            dist = Delta(m)
            dist = Independent(dist, 1)  # Factorized Delta distribution
        elif self.dist == 'Categorical':
            logits = params[0]
            dist = Categorical(logits=logits)
        elif self.dist == 'Bernoulli':
            logits = params[0]
            dist = Bernoulli(logits=logits)
        else:
            dist = None

        return dist


class StochasticLinear2D(StochasticLinear):
    def __init__(self, in_channels, out_channels, dist):
        super(StochasticLinear2D, self).__init__(in_channels, out_channels, dist)
        # Changes the order of the dimension so that the linear layer is applied channel-wise
        self.layer = nn.Sequential(
            Permute(0, 2, 3, 1),
            self.layer,
            Permute(0, 3, 1, 2)
        )

    def forward(self, input):
        dist = super(StochasticLinear2D, self).forward(input)
        dist = Independent(dist, 2) # make the spatial and channel dimensions independent

        return dist


