import re
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import os

CHECKPOINT_FOLDER = 'checkpoints'
SPLIT_TOKEN = '.'
VAR_REGEX = '.*\${([a-z]|[A-Z]|_)+([a-z]|[A-Z]|[0-9]|\.|_)*}.*'


# utilities to flatten and re-inflate the configuration for wandb
def _flatten_config(config, prefix, flat_config):
    for key, value in config.items():
        flat_key = SPLIT_TOKEN.join([prefix, key] if prefix else [key])
        if hasattr(value, 'items'):
            _flatten_config(value, flat_key, flat_config)
        elif not isinstance(value, str):
            flat_config[flat_key] = value
        elif not re.match(VAR_REGEX, value):
            flat_config[flat_key] = value


def flatten_config(config):
    flat_config = {}
    _flatten_config(config, None, flat_config)
    return flat_config


def check_config(config, flat_config):
    for key, value in flat_config.items():
        sub_config = config
        keys = key.split(SPLIT_TOKEN)
        for sub_key in keys[:-1]:
            sub_config = sub_config[sub_key]
        assert sub_config[keys[-1]] == value


class WandbModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        if not wandb.run:
            raise Exception('Wandb has not been initialized. Please call wandb.init first.')
        wandb_dir = wandb.run.dir
        super(WandbModelCheckpoint, self).__init__(dirpath=os.path.join(wandb_dir, CHECKPOINT_FOLDER), *args, **kwargs)