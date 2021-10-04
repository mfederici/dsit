import torch
import torch.nn as nn
from torch.distributions import Distribution
from src.models.base import ConditionalDistribution, RegularizedModel, AdvarsarialModel, RepresentationModel, \
    PredictiveModel


######################################
# Domain Adversarial Neural Networks #
######################################

class DANN(RegularizedModel, AdvarsarialModel, RepresentationModel, PredictiveModel):
    def __init__(self,
                   encoder: ConditionalDistribution,
                   latent_predictor: ConditionalDistribution,
                   discriminator: ConditionalDistribution,
                   beta: float
                   ):
        super(DANN, self).__init__(beta=beta)

        self.generator = nn.ModuleDict({'encoder': encoder, 'latent_predictor': latent_predictor})
        self.discriminator = discriminator

    @property
    def encoder(self):
        return self.generator['encoder']

    @property
    def latent_predictor(self):
        return self.generator['latent_predictor']

    def compute_discriminative_regularization(self, z, e):
        # The regularization loss is the log-probability of the environment when the representation is observed
        p_e_given_z = self.discriminator(z)
        return torch.mean(p_e_given_z.log_prob(e))

    def compute_loss_components(self, data):
        x = data['x']
        y = data['y']
        e = data['e']

        # Encode a batch of data
        q_z_given_x = self.encoder(x)

        # Sample the representation using the re-parametrization trick
        z = q_z_given_x.rsample()

        # Compute the reconstruction distribution
        p_y_given_z = self.latent_predictor(z)

        # The reconstruction loss is the expected negative log-likelihood of the input
        #  - E[log p(Y=y|Z=z)]
        rec_loss = - torch.mean(p_y_given_z.log_prob(y))

        # Compute the adversarial regularization loss
        reg_loss = self.compute_discriminative_regularization(z, e)

        return {'reconstruction': rec_loss, 'regularization': reg_loss}

    def compute_adversarial_loss(self, data, batch_idx):
        x = data['x']
        e = data['e']

        self.discriminator.train()

        with torch.no_grad():
            # Encode a batch of data
            q_z_given_x = self.encoder(x)

            # Sample the representation using the re-parametrization trick
            z = q_z_given_x.sample()

        loss = - self.compute_discriminative_regularization(z, e)

        return {'loss': loss}

    def predict(self, x, sample_latents=False) -> Distribution:
        # If specified sample the latent distribution
        if sample_latents:
            z = self.encoder(x).sample()
        # Otherwise use the mean of the posterior
        else:
            z = self.encoder(x).mean

        # Compute p(Y|Z=z)
        p_y_given_z = self.latent_predictor(z)

        return p_y_given_z


##################################################
# Conditional Domain Adversarial Neural Networks #
##################################################

class CDANN(DANN):
    def compute_discriminative_regularization(self, z, e, y):
        # The regularization loss is the log-probability of the environment when the representation is observed
        p_e_given_z = self.discriminator(z, y)
        return torch.mean(p_e_given_z.log_prob(e))

    def compute_loss_components(self, data):
        x = data['x']
        y = data['y']
        e = data['e']

        # Encode a batch of data
        q_z_given_x = self.encoder(x)

        # Sample the representation using the re-parametrization trick
        z = q_z_given_x.rsample()

        # Compute the reconstruction distribution
        p_y_given_z = self.latent_predictor(z)

        # The reconstruction loss is the expected negative log-likelihood of the input
        #  - E[log p(Y=y|Z=z)]
        rec_loss = - torch.mean(p_y_given_z.log_prob(y))

        # Compute the adversarial regularization loss
        reg_loss = self.compute_discriminative_regularization(z, e, y)

        return {'reconstruction': rec_loss, 'regularization': reg_loss}

    def compute_adversarial_loss(self, data, batch_idx):
        x = data['x']
        y = data['y']
        e = data['e']

        self.discriminator.train()

        with torch.no_grad():
            # Encode a batch of data
            q_z_given_x = self.encoder(x)

            # Sample the representation using the re-parametrization trick
            z = q_z_given_x.sample()

        loss = - self.compute_discriminative_regularization(z, e, y)

        return {'loss': loss}



