import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from code.utils.wandb_utils import add_config
from pytorch_lightning import seed_everything


@hydra.main(config_path='config', config_name='config.yaml')
def parse(conf: DictConfig):
    if 'seed' in conf:
        seed_everything(conf.seed, workers=True)

    trainer = instantiate(conf.trainer)

    if isinstance(trainer.logger, WandbLogger):
        # add the configuration to the experiment
        add_config(trainer.logger.experiment, conf)

    if not trainer.fast_dev_run:
        for callback_conf in conf.callbacks:
            callback = instantiate(callback_conf)
            trainer.callbacks.append(callback)
        for callback_conf in conf.extra_callbacks:
            callback = instantiate(callback_conf)
            trainer.callbacks.append(callback)

    optimization = instantiate(conf.optimization)

    print(optimization)
    print(trainer.callbacks)

    trainer.fit(optimization)

if __name__ == '__main__':
    parse()