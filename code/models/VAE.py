import torch
from torch.distributions import Distribution
from code.models.base import Model, ConditionalDistribution, MarginalDistribution


class VariationalAutoencoder(Model):
    def __init__(
            self,
            encoder: ConditionalDistribution,
            decoder: ConditionalDistribution,
            prior: MarginalDistribution,
            beta: float
    ):
        '''
        Variational Autoencoder Model
        :param encoder: the encoder architecture
        :param decoder: the decoder architecture
        :param prior: architecture representing the prior
        :param beta: trade-off between regularization and reconstruction coefficient
        '''
        super(VariationalAutoencoder, self).__init__()
        self.beta = beta

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

    def encode(self, x) -> Distribution:
        return self.encoder(x)

    def compute_loss(self, data, data_idx):
        loss_components = self.compute_loss_components(data)

        loss = loss_components['rec_loss'] + self.beta * loss_components['reg_loss']

        return {
            'loss': loss,
            'reconstruction_loss': loss_components['rec_loss'].item(),
            'regularization_loss': loss_components['reg_loss'].item()
        }

    def compute_loss_components(self, data):
        x = data['x']

        # Encode a batch of data
        q_z_given_x = self.encoder(x)

        # Sample the representation using the re-parametrization trick
        z = q_z_given_x.rsample()

        # Compute the reconstruction distribution
        p_x_given_z = self.decoder(z)

        # The reconstruction loss is the expected negative log-likelihood of the input
        #  - E[log p(X=x|Z=z)]
        rec_loss = - torch.mean(p_x_given_z.log_prob(x))

        # The regularization loss is the KL-divergence between posterior and prior
        # KL(q(Z|X=x)||p(Z)) = E[log q(Z=z|X=x) - log p(Z=z)]
        reg_loss = torch.mean(q_z_given_x.log_prob(z) - self.prior().log_prob(z))

        return {'rec_loss': rec_loss, 'reg_loss': reg_loss}

    def reconstruct(self, x, sample_latents=False, sample_output=False):
        # If specified sample the latent distribution
        if sample_latents:
            z = self.encoder(x).sample()
        # Otherwise use the mean of the posterior
        else:
            z = self.encoder(x).mean

        # Compute p(X|Z=z)
        p_x_given_z = self.decoder(z)

        # Return mean or a sample from p(X|Z=z) depending on the sample_output flag
        if sample_output:
            x_rec = p_x_given_z.sample()
        else:
            x_rec = p_x_given_z.mean

        return x_rec

    def sample(self, n_items, sample_output=False):
        # Sample from the prior
        z = self.prior.sample([n_items])

        # Compute p(X|Z=z) for the given sample
        p_x_given_z = self.decoder(z)

        # Return mean or a sample from p(X|Z=z) depending on the sample_output flag
        if sample_output:
            x = p_x_given_z.sample()
        else:
            x = p_x_given_z.mean

        return x
