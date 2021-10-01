import pytorch_lightning as pl


class Optimization(pl.LightningModule):
    def __init__(self):
        super(Optimization, self).__init__()

        self.counters = {
            'iteration': 0,
        }
