from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import os

CHECKPOINT_FOLDER = 'checkpoints'


class WandbModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        if not wandb.run:
            raise Exception('Wandb has not been initialized. Please call wandb.init first.')
        wandb_dir = wandb.run.dir
        super(WandbModelCheckpoint, self).__init__(dirpath=os.path.join(wandb_dir, CHECKPOINT_FOLDER), *args, **kwargs)
