import re
from shutil import copytree
import os
import wandb
import yaml
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch

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
        if sub_config[keys[-1]] != value:
            print(keys[-1], sub_config[keys[-1]], value)
            raise Exception()


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
    # check_config(conf, wandb.config)

    # Copy hydra config into the files folder so that everything is stored
    copytree('.hydra', os.path.join(experiment.dir, 'hydra'))


def load_component(conf, state_dict, component_name):
    if component_name is None:
        component = instantiate(conf)
        component.load_state_dict(state_dict)
        return component
    else:
        new_state_dict = {}
        start_token = None
        for k, v in state_dict.items():
            if component_name in k:
                if start_token is None:
                    start_token = k.split('.')[0]
                else:
                    assert start_token == k.split('.')[0]

                new_state_dict['.'.join(k.split('.')[1:])] = v
                if k.startswith(component_name):
                    recur = False

        if start_token != component_name:
            print(start_token, conf)
            return load_component(conf[start_token], new_state_dict, component_name)
        else:
            component = instantiate(conf[start_token])
            component.load_state_dict(new_state_dict)
            return component


def load_checkpoint(run_path, checkpoint='last.ckpt', component_name=None, download_dir='/tmp', device='cpu'):
    # Get access to the wandb Api
    api = wandb.Api()

    # Retrieve the required run
    run = api.run(run_path)

    # Load the configuration for the run
    conf = OmegaConf.load(run.file('hydra/config.yaml').download(replace=True))

    # Load the configuration for the current device
    device_name = os.environ.get('DEVICE_NAME')
    local_device_conf = OmegaConf.load('config/device/%s.yaml' % device_name)
    if 'device' in local_device_conf:
        local_device_conf = local_device_conf['device']

    # Replace the device with the current one
    conf.device = local_device_conf

    # Donwload the required checkpoint
    file = run.file('checkpoints/%s' % checkpoint).download(download_dir, replace=True)

    # Load it
    checkpoint = torch.load(os.path.join(download_dir, 'checkpoints', checkpoint), map_location=torch.device(device))

    # Instantiate the required architecture and the corresponding weights
    return load_component(conf['optimization'], state_dict=checkpoint['state_dict'], component_name=component_name)

