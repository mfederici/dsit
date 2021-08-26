import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig

from pytorch_lightning.loggers import WandbLogger

from code.utils.wandb_utils import add_config
from pytorch_lightning import seed_everything


@hydra.main(config_path='config', config_name='config.yaml')
def parse(conf: DictConfig):
    if 'seed' in conf:
        seed_everything(conf.seed, workers=True)

    # Instantiate the Pytorch Lightning Trainer
    trainer = instantiate(conf.trainer)

    # On the first thread
    if trainer.global_rank == 0:
        # If using the Weights and Bias Logger
        if isinstance(trainer.logger, WandbLogger):

            # Add the hydra configuration to the experiment using the Wandb API
            add_config(trainer.logger.experiment, conf)

        # Add all the callbacks to the trainer (for logging, checkpointing, ...)
        if not trainer.fast_dev_run:
            for callback_conf in conf.callbacks:
                callback = instantiate(callback_conf)
                trainer.callbacks.append(callback)
            for callback_conf in conf.extra_callbacks:
                callback = instantiate(callback_conf)
                trainer.callbacks.append(callback)

    # Instantiate the optimization procedure, which is a Pytorch Lightning module
    optimization = instantiate(conf.optimization)

    # print the model structure
    print(optimization)

    # Call the Pytorch Lightning training loop
    trainer.fit(optimization)


if __name__ == '__main__':
    parse()
