from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from torch.optim import Adam
import pytorch_lightning as pl
from code.models.base import Model
from code.utils.schedulers import Scheduler


class Optimization(pl.LightningModule):
    def __init__(self):
        super(Optimization, self).__init__()

        self.counters = {
            'iteration': 0,
        }


class AdamBatchOptimization(Optimization):
    def __init__(self,
                 model: Model,              # The model to optimize
                 data: dict,                # The dictionary of Datasets defined in the previous 'Data' section
                 num_workers: int,          # Number of workers for the data_loader
                 batch_size: int,           # Batch size
                 lr: float,                 # Learning rate
                 pin_memory: bool = True    # Flag to enable memory pinning
                 ):
        super(AdamBatchOptimization, self).__init__()

        self.model = model
        self.data = data

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.pin_memory = pin_memory

    # this overrides the pl.LightningModule train_dataloader which is used by the Trainer
    def train_dataloader(self):
        return DataLoader(self.data['train'],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=self.pin_memory)

    def log_components(self, loss_items):
        for name, value in loss_items.items():
            self.log('Train/%s' % name, value)
        for name, value in self.counters.items():
            self.log(name, value)

    def training_step(self, data, data_idx) -> STEP_OUTPUT:
        loss_items = self.model.compute_loss(data, data_idx)
        self.log_components(loss_items)
        self.counters['iteration'] += 1
        return loss_items

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)


class AdamBatchRegularizedOptimization(AdamBatchOptimization):
    def __init__(self, beta_scheduler: Scheduler, **params):
        super(AdamBatchRegularizedOptimization, self).__init__(**params)
        self.beta_scheduler = beta_scheduler

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        beta = self.beta_scheduler(self.counters['iterations'])
        self.model.beta = beta
        self.log('Train/beta', beta)
        return super(AdamBatchRegularizedOptimization, self).training_step(batch, batch_idx)
