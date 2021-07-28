from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader
from torch.optim import Adam
import pytorch_lightning as pl
from code.models.base import Model


class AdamBatchOptimization(pl.LightningModule):
    def __init__(self,
                 model: Model,
                 data: dict,
                 num_workers: int,
                 batch_size: int,
                 lr: float
                 ):
        super(AdamBatchOptimization, self).__init__()

        self.model = model
        self.data = data

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr

    def train_dataloader(self):
        return DataLoader(self.data['train'],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def training_step(self, data, data_idx) -> STEP_OUTPUT:
        return self.model.compute_loss(data, data_idx)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)
