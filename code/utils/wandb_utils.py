import re
from shutil import copytree
import os
import wandb
import yaml
from omegaconf import OmegaConf

SPLIT_TOKEN = '.'
VAR_REGEX = '.*\${([a-z]|[A-Z]|_)+([a-z]|[A-Z]|[0-9]|\.|_)*}.*'
KEYS_TO_EXCLUDE_FROM_CONFIG =['device', 'run', 'logger', 'callbacks', 'extra_callbacks']


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


# Add the configuration to the experiment
def add_config(experiment, conf):
    # Create a dictionary with the unresolved configuration
    unresolved_config = dict(yaml.safe_load(OmegaConf.to_yaml(conf, resolve=False)))
    # Ignore some irrelevant configuration
    unresolved_config = {k: v for k, v in unresolved_config.items() if not (k in KEYS_TO_EXCLUDE_FROM_CONFIG)}
    # Flatten the configuration
    flat_config = flatten_config(unresolved_config)

    # Update the configuration
    experiment.config.update(flat_config)

    # Check for inconsistencies
    check_config(conf, wandb.config)

    # Copy hydra config into the files folder so that everything is stored
    copytree('.hydra', os.path.join(experiment.dir, 'hydra'))
