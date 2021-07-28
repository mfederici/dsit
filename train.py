import hydra
from hydra.utils import instantiate
from shutil import copytree
import os
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning.loggers import WandbLogger
import wandb
import yaml
from utils.wandb_utils import flatten_config, check_config
from pytorch_lightning import seed_everything


@hydra.main(config_path='config', config_name='config.yaml')
def parse(conf: DictConfig):
    if 'seed' in conf:
        seed_everything(conf.seed, workers=True)

    trainer = instantiate(conf.trainer)

    logger = trainer.logger

    if isinstance(logger, WandbLogger):
        experiment = logger.experiment

        # Create a dictionary with the unresolved configuration
        unresolved_config = dict(yaml.safe_load(OmegaConf.to_yaml(conf, resolve=False)))
        # Ignore some configuration
        unresolved_config = { k: v for k, v in unresolved_config.items() if not (k in
                                                                                 ['device',
                                                                                  'run',
                                                                                  'logger',
                                                                                  'callbacks',
                                                                                  'extra_callbacks'])}
        # Flatten the configuration
        flat_config = flatten_config(unresolved_config)

        # Update the configuration
        experiment.config.update(flat_config)

        # Check for inconsistencies
        check_config(conf, wandb.config)

        # Copy hydra config into the files folder
        copytree('.hydra', os.path.join(experiment.dir, 'hydra'))

    if not trainer.fast_dev_run:
        for callback_conf in conf.callbacks:
            callback = instantiate(callback_conf)
            trainer.callbacks.append(callback)
        for callback_conf in conf.extra_callbacks:
            callback = instantiate(callback_conf)
            trainer.callbacks.append(callback)

    optimization = instantiate(conf.optimization)

    print(optimization)

    trainer.fit(optimization)

if __name__ == '__main__':
    parse()