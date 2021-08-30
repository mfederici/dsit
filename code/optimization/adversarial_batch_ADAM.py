from pytorch_lightning.utilities.types import STEP_OUTPUT
from code.optimization import AdamBatchRegularizedOptimization
from torch.optim import Adam


class AdversarialAdamBatchRegularizedOptimization(AdamBatchRegularizedOptimization):
    def __init__(self, disc_lr: float, n_adversarial_steps: int, **params):
        super(AdversarialAdamBatchRegularizedOptimization, self).__init__(**params)
        self.disc_lr = disc_lr
        self.n_adversarial_steps = n_adversarial_steps
        self.counters['adversarial_iteration'] = 0

    def training_step(self, batch, batch_idx, optimizer_idx) -> STEP_OUTPUT:
        # Generator
        if optimizer_idx == 1:
            return super(AdamBatchRegularizedOptimization, self).training_step(batch, batch_idx)
        # Discriminator
        elif optimizer_idx == 0:
            loss_items = self.model.compute_adversarial_loss(batch, batch_idx)
            self.log('Train/AdversarialLoss', loss_items['loss'])
            self.counters['adversarial_iteration'] += 1
            return loss_items

    def configure_optimizers(self):
        gen_opt = Adam(self.model.generator.parameters(), lr=self.lr)
        dis_opt = Adam(self.model.discriminator.parameters(), lr=self.disc_lr)
        return (
            {'optimizer': dis_opt, 'frequency': self.n_adversarial_steps},
            {'optimizer': gen_opt, 'frequency': 1}
        )
