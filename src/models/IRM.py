import torch
from torch.distributions import Distribution
import torch.autograd as autograd
import torch.nn.functional as F

from src.models.base import ConditionalDistribution, RegularizedModel, PredictiveModel


###############################
# Invarince Risk Minimization #
###############################

class IRM(RegularizedModel, PredictiveModel):
    def __init__(self,
                 predictor: ConditionalDistribution,
                 beta: float,
                 n_envs: int = 2
                 ):
        super(IRM, self).__init__(beta=beta)

        self.predictor = predictor
        self.n_envs = n_envs

    def compute_reg_loss(self, logits, y, e):
        scale = torch.tensor(1.).to(logits.device).requires_grad_()
        penalty = 0
        for i in range(self.n_envs):
            logits_e = logits[e == i]
            y_e = y[e == i]
            loss_1 = F.cross_entropy(logits_e[::2] * scale, y_e[::2].long())
            loss_2 = F.cross_entropy(logits_e[1::2] * scale, y_e[1::2].long())
            grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
            grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
            penalty += torch.mean(grad_1 * grad_2)
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
        reg_loss = self.compute_reg_loss(q_y_given_x.logits, y, e)

        return {'reconstruction': torch.mean(rec_loss), 'regularization': reg_loss}

    def predict(self, x) -> Distribution:
        return self.predictor(x)


