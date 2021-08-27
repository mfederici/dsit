import torch
from torch.distributions import Distribution

from code.architectures.utils import OneHot
from code.models.base import ConditionalDistribution, MarginalDistribution, RegularizedModel


#####################################
# Variance-based Risk Extrapolation #
#####################################

class VREx(RegularizedModel):
    def __init__(self,
                   predictor: ConditionalDistribution,
                   beta: float,
                   n_envs: int = 2,
                   use_std: bool = True
                   ):
        super(VREx, self).__init__(beta=beta)

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
        q_y_given_x = self.prefictor(x)

        # The reconstruction loss is the expected negative log-likelihood of the input
        #  - E[log p(Y=y|Z=z)]
        rec_loss = - q_y_given_x.log_prob(y)

        # Compute_the regularization
        reg_loss = self.compute_reg_loss(rec_loss, e)

        return {'reconstruction': torch.mean(rec_loss), 'regularization': reg_loss}

    def predict(self, x) -> Distribution:
        return self.predictor(x)


