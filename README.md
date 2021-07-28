# Installing
## Conda environment
Create a new `representation-learning` environment using conda:
```shell
conda env create -f environment.yml
```
and activate it:
```shell
conda activate frame
```

## Naming the device
Assign a name to the device that will be running the experiments and make sure there is a corresponding device in
`config/device` contains a configuration file with the same name:
```shell
export DEVICE_NAME=<DEVICE_NAME>
```
You can add the previous export to the `.bashrc` file to avoid running it every time a new session is created.

The corresponding `config/device/<DEVICE_NAME>.yaml` device configuration file contains device-specific information 
regarding hardware and paths.
Here we report an example for a device configuration:
```yaml
# Example of the content of config/device/<DEVICE_NAME>
data_root: /ssdstore/data
big_data_root: /hddstore/data
experiments_root: /hddstore
download_files: true
num_workers: 32
gpus: 4
```
This setup allow easy deployment of the same code on different machines since all the hardware-dependent configuration 
is grouped into the device `.yaml` configuration file. 

## Weights & Bias logging
For [Weights & Bias](www.wandb.ai) logging run:
```shell
wandb init
```
and login with your credentials. This step is optional since [TensorBoard](https://www.tensorflow.org/tensorboard) 
logging is also [implemented](#tensorboard_loggging).

# Running experiments
The CLI for training models is based on [Hydra](www.hydra.cc). See this 
[simple example](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli)
and [basic grammar for parameter override](https://hydra.cc/docs/advanced/override_grammar/basic) for further information
about Hydra.

To run an experiment use:
```shell
python train.py +experiment=<EXPERIMENT_NAME> <OTHER_OPTIONS>
```
As an example, the following command can be used to run a Variational Autoencoder on the MNIST dataset for 20 epochs
with the value of the hyper-parameter `beta` set to `0.1`.
```shell
python train.py +experiment=VAE_MNIST +trainer.max_epochs=20 params.beta=0.1
```
See the [Experiment Definition](#defining-new-experiments) for further information regarding the experimental setup

## Tensorboard Logging
default logging is with wandb, but it is possible to switch to tensorboard with
```shell
python train.py +experiment=<EXPERIMENT_NAME> logging=tensorboard
```
Alternative loggers can be defined in the `config/logging` configuration `.yaml` files.

## Sweeps with Weights and Bias
The `train.py` script is defined to be compatible with [wandb hyper-parameters sweeps](https://docs.wandb.ai/guides/sweeps).

Each sweep definition can directly access the properties and hyper-parameters defined in the configuration files.
The [following file](sweeps/VAE_MNIST.yml) reports an example for the [MNIST Variational Autoencoder experiment](config/experiment/MNIST_VAE.yaml):
```yaml
program: train.py
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - +experiment=MNIST_VAE       # The experiment to launch
  - +trainer.max_epochs=10      # Training for 10 epochs
  - ${args_no_hyphens}          # parameters from the sweep (params.beta in this case)
method: bayes                   # bayesian optimization
metric:
  goal: maximize
  name: ELBO/Validation         # Metric logged and defined in the config/experient/MNIST_VAE.yaml file
parameters:
  params.beta:                  # The hyper-parameter beta
    distribution: log_uniform   # will be sampled uniformly in the log space
    min: -20                    # from e^-20
    max: 2                      # to e^2
```
The sweep can be created by running:
```shell
wandb sweep sweeps/VAE_MNIST.yaml
```
which will return the corresponding `<SWEEP_ID>.
Agents will be then started with:
```shell
wandb agent <SWEEP_ID>
```
# The run configuration
The configuration for each run is composed by the following main components:
- [**data**](#data): the data used for training the models. See section for further 
  information.
- [**model**](#Models) : the model to train. Each model must implement all the architecture and data-agnostic logic regarding 
  the loss computation (e.g. `VAE`, `GAN`, `VIB`, ...). Each model is an instance of a 
  [Pytorch Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).
- [**optimization**](#optimization-procedures): the procedure used for optimizing the model. Definition on how the model is updated by the optimizer 
  (e.g. standanrd step update, adversarial training, joint training of two models, optimizer type, batch-creation procedure). Each optimization procedure is an 
  instance of a [Lighning Module](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html).
- [**architectures**](#architectures): definition of the data-specific architectures used by the model. 
  (e.g. `Encoder`, `Decoder` and `Prior`  for a VAE). Note that different model could make use of the same architectures
  (e.g. both Variational Autoencoder and Variational Information Bottleneck can use the same 'Encoder'). Architectures
  are also instances of Pytorch Modules.
- [**params**](#hyper-parameters): collection of the model, architecture, optimization and data hyper-parameters (e.g. 
  number of layers, learning rate, batch size, regularization strength, ...). This design allows for easy definition of 
  [sweeps and hyper-parameter tuning](https://docs.wandb.ai/guides/sweeps).
- [**callbacks**](#callbacks): the callbacks called during training. Different callbacks can be used for logging, evaluation
  , model checkpointing or early stopping. 
  See [the corresponding documentation](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html) 
  for further details. Note that callbacks are fully optional.
- [**device**](#device): Definition of the hardware-specific parameters (such as paths, CPU cores, number of GPUs)
- [**logging**](#logging): Definition of the logging procedure. Both TensorBoard and Weights & Bias are supported.
- [**trainer**](#trainer): Extra parameters passed to the [Ligthning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)

While data, model, architectures, optimization procedure, parameters and callbacks are experiment-specific, device, logging 
and training define global properties of the device on which the experiments are running, the logging and training 
procedures respectively.

## Defining new Experiments
Each experiment `.yaml` configuration file contains a definition of data, model, architectures, optimization procedure, 
hyper-parameters and callbacks. 
Here we report an example for the 
[Variational Autoencoder Model trained on the MNIST dataset](config/experiment/MNIST_VAE.yaml).

First we refer to the `VAE` model, `MNIST` dataset and `batch_ADAM` optimization procedure defined in `config/model/VAE.yaml`,
`config/data/MNIST.yaml`, and `config/optimization/batch_ADAM.yaml` respectively:
```yaml
# @package _global_
defaults:
  - /model: VAE
  - /data: MNIST
  - /optimization: batch_ADAM
```
The [VAE model](config/model/VAE.yaml) requires the definition of an `encoder`, `decoder` and `prior` architectures: 
```yaml
architectures:
  prior:
    _target_: code.architectures.base.DiagonalNormal
    z_dim: ${params.z_dim}
  decoder:
    _target_: code.architectures.MNIST.Decoder
    z_dim: ${params.z_dim}
    layers: ${params.decoder_layers}
  encoder:
    _target_: code.architectures.MNIST.Encoder
    layers: ${params.encoder_layers}
    z_dim: ${params.z_dim}
```
The `_target_` key contains references to the corresponding python classes, while the other values are passed to the 
`__init__()` constructor on initialization.

Note that instead of writing the value of the hyper-parameters (such as the number of latents `z_dim`) directly in the 
architecture definition, we refer to the `params` section (e.g. `${params.z_dim}`) so that all the hyper-parameters of 
model, architectures and optimization procedure are grouped together:
```yaml
params:
  z_dim: 64
  beta: 0.5
  lr: 1e-3
  batch_size: 128
  encoder_layers: [ 1024, 128 ]
  decoder_layers: [ 128, 1024 ]
```
Lastly, a list of callbacks defines all the evaluation metrics that are logged during training:
```yaml
callbacks:
  - _target_: utils.callbacks.EvaluationCallback
    name: ImageReconstruction/Validation
    evaluate_every: 10 seconds
    evaluator:
      _target_: code.evaluation.reconstruction.ImageReconstructionQualitativeEvaluation
      evaluate_on: valid
      n_pictures: 10
      sample_images: False
      sample_latents: False
  - _target_: utils.callbacks.EvaluationCallback
    name: ELBO/Validation
    evaluate_every: 10 seconds
    evaluator:
      _target_: code.evaluation.elbo.ELBOEvaluation
      evaluate_on: valid
      n_samples: 2048
  - _target_: utils.callbacks.EvaluationCallback
    name: ELBO/Train
    evaluate_every: 10 seconds
    evaluator:
      _target_: code.evaluation.elbo.ELBOEvaluation
      evaluate_on: train
      n_samples: 2048
```
Further details regarding the `EvaluationCallback` utility class and the evaluation procedures can be found in the 
[corresponding section](#callbacks)

Further details regarding the aforementioned components can be found in the following sections
# Creating new Implementations
## Data
## Models
## Architectures 
## Optimization procedures
## Callbacks
### Evaluation
## Device
## Logging
