import torch
from torch.distributions import Distribution

from src.architectures.utils import OneHot
from src.models.base import ConditionalDistribution, RegularizedModel, PredictiveModel


#####################################
# Variance-based Risk Extrapolation #
#####################################

class VREx(RegularizedModel, PredictiveModel):
    def __init__(self,
                   predictor: ConditionalDistribution,
                   beta: float,
                   n_envs: int = 2,
                   use_std: bool = True,
                   sum_batch_penalty: bool = True
                   ):
        super(VREx, self).__init__(beta=beta)

        self.predictor = predictor
        self.n_envs = n_envs
        self.use_std = use_std
        self.one_hot = OneHot(n_envs)
        self.sum_batch_penalty = sum_batch_penalty

    def compute_reg_loss(self, y_rec_loss, e):

        # Long to one hot encoding
        one_hot_e = self.one_hot(e.long())

        # Environment variance penalty

        # Count how many examples for each one of the environments
        e_sum = one_hot_e.sum(0)

        # Compute the total loss per environment
        env_loss = (y_rec_loss.unsqueeze(1) * one_hot_e).sum(0)

        # Compute the mean loss per environment (with at least one occurrence)
        env_loss[e_sum > 0] = env_loss[e_sum > 0] / e_sum[e_sum > 0]

        # The variance is the expected mean of the difference E[(l(e,x,y)- l(e))**2]
        loss_variance = ((env_loss - env_loss[e_sum > 0].mean()) ** 2)[e_sum > 0].mean()

        if self.sum_batch_penalty:
            batch_size = e.shape[0]
            loss_variance *= batch_size

        if not self.use_std:
            penalty = loss_variance
        else:
            penalty = loss_variance ** 0.5

        return penalty

    def compute_loss_components(self, data):
        x = data['x']
        y = data['y']
        e = data['e']

        # Encode a batch of data
        q_y_given_x = self.predictor(x)

        # The reconstruction loss is the expected negative log-likelihood of the input
        #  - E[log p(Y=y|Z=z)]
        rec_loss = - q_y_given_x.log_prob(y)

        # Compute_the regularization
        reg_loss = self.compute_reg_loss(rec_loss, e)

        return {'reconstruction': torch.mean(rec_loss), 'regularization': reg_loss}

    def predict(self, x) -> Distribution:
        return self.predictor(x)


