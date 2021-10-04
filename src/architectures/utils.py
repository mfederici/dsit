import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, Beta, Categorical, Bernoulli, Distribution
from pyro.distributions import Delta, TransformModule, constraints
from torch.nn.functional import softplus
import numpy as np
from torch.nn.utils import spectral_norm as sn


# Create simple layer stacks with relu activations
def make_stack(layers, dropout=0.0, spectral_norm=False):
    nn_layers = []
    for i in range(len(layers)-1):
        layer = nn.Linear(layers[i], layers[i+1])
        if spectral_norm:
            layer = sn(layer)
        nn_layers.append(layer)
        if i < len(layers)-2:
            if dropout > 0:
                nn_layers.append(nn.Dropout(dropout))
            nn_layers.append(nn.ReLU(True))

    return nn_layers


def make_cnn_stack(layers, dropout=0.0):
    cnn_layers = []
    for i in range(len(layers)):
        cnn_layers.append(nn.Conv2d(**layers[i]))
        if i < len(layers)-2:
            if dropout > 0:
                cnn_layers.append(nn.Dropout2d(dropout))
            cnn_layers.append(nn.ReLU(True))

    return cnn_layers


def make_cnn_deconv_stack(layers):
    cnn_layers = []
    for i in range(len(layers)):
        cnn_layers.append(nn.ConvTranspose2d(**layers[i]))
        if i < len(layers)-2:
            cnn_layers.append(nn.ReLU(True))

    return cnn_layers


class Flatten(TransformModule):
    def __init__(self):
        super(Flatten, self).__init__()
        self.shape = None

    def forward(self, input):
        if self.shape is None:
            self.shape = input.shape[1:]

        assert self.shape == input.shape[1:]
        return input.view(input.shape[0], -1)

    def log_abs_det_jacobian(self, x, y):
        return 1

    def _call(self, x):
        return self.forward(x)

    def _inverse(self, output):
        return output.view(-1, *self.shape)


class Reshape(TransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    sign = +1

    def __init__(self, in_shape, out_shape, contiguous=False):
        super(Reshape, self).__init__()
        assert np.prod(in_shape) == np.prod(out_shape)
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.contiguous = contiguous

    def forward(self, input):
        if self.contiguous:
            return input.reshape(-1, *self.out_shape)
        else:
            return input.view(-1, *self.out_shape)

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros(
            x.size()[:-1], dtype=x.dtype, layout=x.layout, device=x.device
        )

    def _call(self, x):
        return self.forward(x)

    def _inverse(self, output):
        if self.contiguous:
            return output.reshape(-1, *self.in_shape)
        else:
            return output.view(-1, *self.in_shape)


class SequentialModel(nn.Module):
    def __init__(self, models):
        super(SequentialModel, self).__init__()
        for model in models:
            assert isinstance(model, nn.Module)

        self.models = nn.ModuleList(models)

    def forward(self, x):
        for model in self.models[:-1]:
            x = model(x)
            if isinstance(x, Distribution):
                x = x.rsample()
        return self.models[-1](x)


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

    def extra_repr(self) -> str:
        return "distribution=%s" % self.dist


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


