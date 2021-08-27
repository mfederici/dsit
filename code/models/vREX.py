import torch
from torch.distributions import Distribution

from code.architectures.utils import OneHot
from code.models.base import ConditionalDistribution, MarginalDistribution, RegularizedModel


######################################
# Variational Information Bottleneck #
######################################

class VREx(RegularizedModel):
    def __init__(self,
                   encoder: ConditionalDistribution,
                   predictor: ConditionalDistribution,
                   beta: float,
                   n_envs: int = 2,
                   use_std: bool = True
                   ):
        super(VREx, self).__init__(beta=beta)

        self.encoder = encoder
        self.predictor = predictor
        self.n_envs = n_envs
        self.use_std = use_std
        self.one_hot = OneHot(n_envs)

    def compute_reg_loss(self, y_rec_loss, e):

        # Long to one hot encoding
        one_hot_e = self.one_hot(e.long())

        # Environment variance penalty
        e_sum = one_hot_e.sum(0)
        env_loss = (y_rec_loss.unsqueeze(1) * one_hot_e).sum(0)
        env_loss[e_sum > 0] = env_loss[e_sum > 0] / e_sum[e_sum > 0]
        loss_variance = ((env_loss - env_loss[e_sum > 0].mean()) ** 2)[e_sum > 0].mean()

        if not self.use_std:
            return loss_variance
        else:
            return loss_variance ** 0.5

    def compute_loss_components(self, data):
        x = data['x']
        y = data['y']
        e = data['e']

        # Encode a batch of data
        q_z_given_x = self.encoder(x)

        # Sample the representation using the re-parametrization trick
        z = q_z_given_x.rsample()

        # Compute the reconstruction distribution
        p_y_given_z = self.predictor(z)

        # The reconstruction loss is the expected negative log-likelihood of the input
        #  - E[log p(Y=y|Z=z)]
        rec_loss = - p_y_given_z.log_prob(y)

        # The regularization loss is the KL-divergence between posterior and prior
        # KL(q(Z|X=x)||p(Z)) = E[log q(Z=z|X=x) - log p(Z=z)]
        reg_loss = self.compute_reg_loss(rec_loss, e)

        return {'reconstruction': torch.mean(rec_loss), 'regularization': reg_loss}

    def encode(self, x) -> Distribution:
        return self.encoder(x)

    def predict(self, x, sample_latents=False) -> Distribution:
        # If specified sample the latent distribution
        if sample_latents:
            z = self.encoder(x).sample()
        # Otherwise use the mean of the posterior
        else:
            z = self.encoder(x).mean

        # Compute p(Y|Z=z)
        p_y_given_z = self.predictor(z)

        return p_y_given_z


