import torch
from torch.distributions import Distribution
from code.models.base import ConditionalDistribution, MarginalDistribution, RegularizedModel


######################################
# Variational Information Bottleneck #
######################################

class VariationalInformationBottleneck(RegularizedModel):
    def __init__(self,
                   encoder: ConditionalDistribution,
                   predictor: ConditionalDistribution,
                   prior: MarginalDistribution,
                   beta: float
                   ):
        super(VariationalInformationBottleneck, self).__init__(beta=beta)

        self.encoder = encoder
        self.predictor = predictor
        self.prior = prior

    def compute_loss_components(self, data):
        x = data['x']
        y = data['y']

        # Encode a batch of data
        q_z_given_x = self.encoder(x)

        # Sample the representation using the re-parametrization trick
        z = q_z_given_x.rsample()

        # Compute the reconstruction distribution
        p_y_given_z = self.predictor(z)

        # The reconstruction loss is the expected negative log-likelihood of the input
        #  - E[log p(Y=y|Z=z)]
        rec_loss = - torch.mean(p_y_given_z.log_prob(y))

        # The regularization loss is the KL-divergence between posterior and prior
        # KL(q(Z|X=x)||p(Z)) = E[log q(Z=z|X=x) - log p(Z=z)]
        reg_loss = torch.mean(q_z_given_x.log_prob(z) - self.prior().log_prob(z))

        return {'reconstruction': rec_loss, 'regularization': reg_loss}

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


