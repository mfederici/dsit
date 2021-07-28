from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import LightningLoggerBase

class Evaluation:
    def evaluate(self, optimization: LightningModule):
        raise NotImplemented()
