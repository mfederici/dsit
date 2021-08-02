# A Scalable and Re-usable Framework for Deep Learning
The goal of this repository is to show an example of code implementation and usage of the Weights & Biases, 
Pytorch Lightning, Hydra and their interaction. This implementation is inspired by the [Reproducible Deep Learning PhD 
course](https://www.sscardapane.it/teaching/reproducibledl/) from the at Sapienza University (Rome). 

The main design principle driving this project are:
- **Modularity**: each part of the training/evaluation/logging procedure is implemented as an independent block with 
pre-defined interactions.
- **Extensibility**: the framework can be easily extended to include new experiments/models/logging procedures,
architectures and datasets
- **Clarity/Readability**: each model contains only the data-agnostic and architecture agnostic-logic.

Main features:
- Get all the perks of a Pytorch [Ligthning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)
- Use the Weights & Bias [sweep tool](https://docs.wandb.ai/guides/sweeps) 
to easily define hyper-parameters search and launch multiple agents across different devices.
- Easily handle complex configuration thanks to the powerful [Hydra configuration management](https://hydra.cc/docs/intro/).
- No need to write any training/configuration script.

As a general concept, the framework is designed to that each piece can be maximally re-used:
the same model can be used with different architectures and dataset, the same architectures and evaluation procedures 
can be used with different models, and the same optimization procedure can be used for different models without the need to re-write any code.




# Installing
## Conda environment
Create a new `frame` environment using conda:
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

The corresponding `config/device/<DEVICE_NAME>.yaml` configuration file contains device-specific information 
regarding hardware and paths.
Here we report an example for a device configuration:
```yaml
# Example of the content of config/device/<DEVICE_NAME>
data_root: /ssdstore/data             # Path in which small dataset are stored
big_data_root: /hddstore/data         # Path for the big ones
experiments_root: /hddstore           # Location in which the experiments are stored
download_files: true                  # Flag to specify if the device allows for downloading files
num_workers: 32                       # Number of workers spawned for data-loading

# @package _global_
trainer:      # Accessing and changing the parameters of the Pytorch Ligthning trainer
  gpus: 4     # Number of gpus used for training
```
With this setup, the same code on different machines since all the hardware-dependent configuration 
is grouped into the device `.yaml` configuration file. 

## Weights & Bias logging
For [Weights & Bias](www.wandb.ai) logging run:
```shell
wandb init
```
and login with your credentials. This step is optional since [TensorBoard](https://www.tensorflow.org/tensorboard) 
logging [is also implemented](#tensorboard_loggging).

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
The default logging uses Weights & Biases, but it is possible to switch to TensorBoard with
```shell
python train.py +experiment=<EXPERIMENT_NAME> logging=tensorboard
```
Alternative loggers can be defined in the `config/logging` configuration `.yaml` files.

## Sweeps with Weights and Bias
The `train.py` script is defined to be compatible with [wandb hyper-parameters sweeps](https://docs.wandb.ai/guides/sweeps).

Each sweep definition can directly access the properties and hyper-parameters defined in the configuration files.
The [following file](sweeps/VAE_MNIST.yml) reports an example sweep for the [MNIST Variational Autoencoder experiment](config/experiment/MNIST_VAE.yaml):
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
Agents can then be started with:
```shell
wandb agent <WANDB_USER>/<WANDB_PROJECT>/<SWEEP_ID>
```
# The run configuration
The configuration for each run is composed by the following main components:
- [**data**](#data): the data used for training the models. See section for further 
  information.
- [**model**](#Models) : the model to train. Each model must implement the logic regarding the loss computation 
- (e.g. `VAE`, `GAN`, `VIB`, ...) and functionalities (e.g. `sample`, `reconstruct`, `classify`,...) in architecture and 
  data-agnostic fashion. Each model is an instance of a [Pytorch Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).
- [**optimization**](#optimization-procedures): the procedure used for optimizing the model. Definition on how the model is updated by the optimizer 
  (e.g. standanrd step update, adversarial training, joint training of two models, optimizer type, batch-creation procedure). Each optimization procedure is an 
  instance of a [Lighning Module](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html).

- [**params**](#hyper-parameters): collection of the model, architecture, optimization and data hyper-parameters (e.g. 
  number of layers, learning rate, batch size, regularization strength, ...). This design allows for easy definition of 
  [sweeps and hyper-parameter tuning](https://docs.wandb.ai/guides/sweeps).
- [**trainer**](#trainer): Extra parameters passed to the [Ligthning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)
- [**callbacks**](#callbacks): the callbacks called during training. Different callbacks can be used for logging, 
[evaluation](#evaluation), [model checkpointing](#checkpoints) or early stopping. 
  See [the corresponding documentation](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html) 
  for further details. Note that callbacks are fully optional.
- [**device**](#device): Definition of the hardware-specific parameters (such as paths, CPU cores, number of GPUs)
- [**logger**](#loggers): Definition of the logging procedure. Both TensorBoard and Weights & Bias are supported.

While `data`, `model`, `architectures`, `optimization procedure`, `parameters` and `callbacks` are experiment-specific, 
`device`, `logging` and `trainer` define global properties of the device on which the experiments are running, 
the logging and training parameters respectively.

## Defining new Experiments
Each experiment `.yaml` configuration file contains a definition of data, model, architectures, optimization procedure, 
hyper-parameters and callbacks. 
Here we report an example for the 
[Variational Autoencoder Model trained on the MNIST dataset](config/experiment/MNIST_VAE.yaml).

First we refer to `MNIST` dataset and `batch_ADAM` optimization procedure defined in
the [data](config/data/MNIST.yaml), and [optimization](config/optimization/batch_ADAM.yaml) configuration files respectively:
```yaml
# @package _global_
defaults:
  - /data: MNIST
  - /optimization: batch_ADAM
```
in which `# @package _global_` line is used to specify that the specified keys are global, while `defaults` specifies 
the values for `data` and `optimization` procedures respectively. Further information regarding
configuration packages and overrides can be found [here](https://hydra.cc/docs/advanced/overriding_packages).

The [VAE model](code/models/unsupervised/VAE.py) requires the definition of an `encoder`, `decoder` and `prior` architectures: 
```yaml
model:
  _target_: code.models.unsupervised.VariationalAutoencoder # class defining the VAE model
  prior:                                                    # Prior distribution
    _target_: code.architectures.base.DiagonalNormal        # class defining a Normal distribution with diagonal covariance
    z_dim: ${params.z_dim}
  encoder:                                                  # Encoder architecture
    _target_: code.architectures.MNIST.Encoder              
    layers: ${params.encoder_layers}
    z_dim: ${params.z_dim}
  decoder:                                                  # Decoder architecture
    _target_: code.architectures.MNIST.Decoder              
    z_dim: ${params.z_dim}
    layers: ${params.decoder_layers}
  beta: ${params.beta}
```
The `_target_` key contains references to Python classes, while the other values are passed to the 
`__init__()` constructor on initialization (e.g. `Encoder(layers, z_dim)` is called when instantiating the encoder architecture).

Note that instead of writing the value of the hyper-parameters (such as the number of latents `z_dim` or regularization 
strength `beta`) directly in the architecture definition, we refer to the `params` section (e.g. `${params.z_dim}`, 
`${params.beta}`) so that all the hyper-parameters of model, architectures and optimization procedure are grouped together:
```yaml
params:                               # List of hyper-parameters
  z_dim: 64                           # Number of latent dimensions
  beta: 0.5                           # KL regularization strength
  encoder_layers: [ 1024, 128 ]       # List of hidden layers for the encoder
  decoder_layers: [ 128, 1024 ]       # and decoder architectures
  lr: 1e-3                            # Learning rate
  batch_size: 128                     # Batch size
```
Lastly, a list of callbacks defines all the evaluation metrics that are logged during training:
```yaml
callbacks:
  # Logging the validation image reconstructions 
  - _target_: code.callbacks.EvaluationCallback      # Utility callback for evaluation that logs every 'evaluate_every'
    name: ImageReconstruction/Validation              # Name reported in the log
    evaluate_every: 60 seconds                        # Evaluation time (in seconds, minutes, hours, iterations or epochs)
    evaluator:
      _target_: code.evaluation.image.ImageReconstructionEvaluation # Class defining the evaluation
      evaluate_on: valid
      n_pictures: 10
      sample_images: False
      sample_latents: False
      
  # Logging the samples of the generative model
  - _target_: code.callbacks.EvaluationCallback
    name: Samples
    evaluate_every: 60 seconds
    evaluator:
      _target_: code.evaluation.image.ImageSampleEvaluation
      evaluate_on: valid
      n_pictures: 10
      
  # Logging the value or the Evidence Lower BOund (ELBO) computed on the validation set
  - _target_: code.callbacks.EvaluationCallback
    name: ELBO/Validation
    evaluate_every: 30 seconds
    evaluator:
      _target_: code.evaluation.elbo.ELBOEvaluation
      evaluate_on: valid
      n_samples: 2048
  
  # Logging the value or the Evidence Lower BOund (ELBO) computed on the train set
  - _target_: code.callbacks.EvaluationCallback
    name: ELBO/Train
    evaluate_every: 30 seconds
    evaluator:
      _target_: code.evaluation.elbo.ELBOEvaluation
      evaluate_on: train
      n_samples: 2048
```
To summarize, the log will consist of the following entries:
- ImageReconstruction/Validation: reconstruction of images from the validation set, logged every 60 seconds
- Samples: images sampled from the prior.
- ELBO/Validation: Evidence LOwer Bound computed on the validation set 
- ELBO/Train: Evidence LOwer Bound computed on the train set
Further details regarding the `EvaluationCallback` utility class and the evaluation procedures can be found in the 
[corresponding section](#callbacks)

Further details regarding the aforementioned components can be found in the following sections
# Creating new Implementations

Adding new models, datasets and architectures to the frameworks requires implementing the code and creating the 
corresponding configuration files.
Here we report the conventions used to define the different components: 

## Data
The datasets definition are collected in the [`data` configuration folder](config/data). Each data object consist
of a dictionary specifying the parameters for the different splits. By default, we consider `train`, `valid`, and `test`
for training, validation and testing purpose respectively. Different keys can be added to the `data` dictionary if necessary:
```yaml
# Content of /config/data/MNIST.yaml. the corresponding keys are added under `data`
train:                                      # Definition of the training set
  _target_: code.data.MNIST.MNISTWrapper    # Class
  root: ${device.data_root}                 # Initialization parameters
  split: train
  download: ${device.download_files}

valid:
  _target_: code.data.MNIST.MNISTWrapper   # Definition of the validation split
  root: ${device.data_root}
  split: valid
  download: ${device.download_files}

test:
  _target_: code.data.MNIST.MNISTWrapper   # Test split
  root: ${device.data_root}
  split: test
  download: ${device.download_files}
```
Note that all the device-dependent parameters (such as the data directory and the flag to enable downloading)
refer to the `${device}` variable. This allows to easily deploy the same model to different devices. Further details can
be found in the [device section](#device).

TorchVision, TorchAudio or other existing datasets class definitions can be referenced directly by specifying
the appropriate `_target_` (e.g. `_target_: torchvision.datasets.MNIST` for the default torchvision MNIST dataset).

The instantiated `data` dictionary is passed to the constructor of the [optimization procedure](#optimization-procedures),
in which the data-loaders are defined.

## Models
The model configuration defines the parameters and architectures used by the specific model (see example reported 
[the previous section](#defining-new-experiments)). The model code is designed to be completely data-agnostic
so that the same logic can be used across different experiments without any re-writing or patch.
Here we report the example code for the `VariationalAutoencoder` [model](code/models/unsupervised/VAE.py):
```python
class VariationalAutoencoder(GenerativeModel, RepresentationLearningModel):
    def __init__(
            self,
            encoder: ConditionalDistribution,
            decoder: ConditionalDistribution,
            prior: MarginalDistribution,
            beta: float
    ):
        '''
        Variational Autoencoder Model
        :param encoder: the encoder architecture
        :param decoder: the decoder architecture
        :param prior: architecture representing the prior
        :param beta: trade-off between regularization and reconstruction coefficient
        '''
        super(VariationalAutoencoder, self).__init__()
       

        # The data-dependent architectures are passed as parameters
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        
        # Store the value of the hyper-parameter beta
        self.beta = beta
    
    # Definition of the procedure to compute reconstruction and regularization loss
    def compute_loss_components(self, data):
        x = data['x']

        # Encode a batch of data
        q_z_given_x = self.encoder(x)

        # Sample the representation using the re-parametrization trick
        z = q_z_given_x.rsample()

        # Compute the reconstruction distribution
        p_x_given_z = self.decoder(z)

        # The reconstruction loss is the expected negative log-likelihood of the input
        #  - E[log p(X=x|Z=z)]
        rec_loss = - torch.mean(p_x_given_z.log_prob(x))

        # The regularization loss is the KL-divergence between posterior and prior
        # KL(q(Z|X=x)||p(Z)) = E[log q(Z=z|X=x) - log p(Z=z)]
        reg_loss = torch.mean(q_z_given_x.log_prob(z) - self.prior().log_prob(z))

        return {'reconstruction': rec_loss, 'regularization': reg_loss}
    
    # Function called by the optimization procedure to compute the loss for one batch
    def compute_loss(self, data, data_idx):
        loss_components = self.compute_loss_components(data)

        loss = loss_components['reconstruction'] + self.beta * loss_components['regularization']

        return {
            'loss': loss,                                               # The 'loss' key is used for gradient computation
            'reconstruction': loss_components['reconstruction'].item(),  # The other keys are returned for logging purposes
            'regularization': loss_components['regularization'].item()
        }
    
    # Function implemented by representation learning models to define the encoding procedure
    def encode(self, x) -> Distribution:
        return self.encoder(x)
    
    # Function implemented by generative models to generate new samples
    def sample(self, sample_shape: torch.Size = torch.Size([]), sample_output=False) -> torch.Tensor:
        # Sample from the prior
        z = self.prior().sample(sample_shape)

        # Compute p(X|Z=z) for the given sample
        p_x_given_z = self.decoder(z)

        # Return mean or a sample from p(X|Z=z) depending on the sample_output flag
        if sample_output:
            x = p_x_given_z.sample()
        else:
            x = p_x_given_z.mean

        return x
```
Modularity and re-usability are the key design principles that allow to re-use and read the model code in a completely
task-agnostic fashion. All the task-dependent code is contained into the parameters (such as `encoder` and `decoder`) 
that are passed to the model.

### Architectures 
Different architectures are implemented in the `code/architectures` folder. Each architecture is designed for a specific 
role (e.g. `Encoder`, `Decoder`, `Predictor`, ...), as a result the same architecture can be used in multiple models.
Since the architecture code is data-dependent, each dataset will correspond to a different set of architectures.

Here we report the example for the `VariationalAutoencoder` `Encoder` on `MNIST`:
```python
INPUT_SHAPE = [1, 28, 28]
N_INPUTS = 28*28
N_LABELS = 10


# Model for q(Z|X)
class Encoder(ConditionalDistribution):
    def __init__(self, z_dim: int, layers: list):
        '''
        Encoder network used to parametrize a conditional distribution
        :param z_dim: number of dimensions for the latent distribution
        :param layers: list describing the layers
        '''
        super(Encoder, self).__init__()

        # Create a stack of layers with ReLU activations as specified
        nn_layers = make_stack([N_INPUTS] + list(layers))

        self.net = nn.Sequential(
            Flatten(),                                      # Layer to flatten the input
            *nn_layers,                                     # The previously created stack
            nn.ReLU(True),                                  # A ReLU activation
            StochasticLinear(layers[-1], z_dim, 'Normal')   # A layer that returns a factorized Normal distribution
        )

    def forward(self, x) -> Distribution:
        # Note that the encoder returns a factorized normal distribution and not a vector
        return self.net(x)
```

## Optimization procedures
The optimization procedure is designed to contain the logic regarding how the model is updated over time.
This includes the definition of optimizers, data-loaders and learning-rate schedulers.
Once again, the optimization procedure is designed to be modular and model-agnostic. Here we report the example
for the a [batch-based training procedure with the ADAM optimizer](code/optimization/batch_ADAM.py):
```python
# Each optimization procedure is a pytorch lightning module
class AdamBatchOptimization(pl.LightningModule):
    def __init__(self,
                 model: Model,          # The model to optimize
                 data: dict,            # The dictionary of Datasets defined in the previous 'Data' section
                 num_workers: int,      # Number of workers for the data_loader
                 batch_size: int,       # Batch size
                 lr: float,              # Learning rate
                 pin_memory: bool=True  # Flag to enable memory pinning
                 ):
        super(AdamBatchOptimization, self).__init__()

        self.model = model
        self.data = data

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.pin_memory = pin_memory

    # this overrides the pl.LightningModule train_dataloader which is used by the Trainer
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data['train'],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=self.pin_memory)

    # The training step simply returns the computation from the model
    def training_step(self, data, data_idx) -> STEP_OUTPUT:
        return self.model.compute_loss(data, data_idx)
    
    # Instantiate the Adam optimizer passing the model trainable parameters
    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)
```
Each optimization procedure is a Pytorch Ligthning module, therefore it is possible to extend all the corresponding
 [functions](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html) for customized 
training/data loading.

The corresponding [configuration file](config/optimization/batch_ADAM.yaml) simply defines the references for the
parameters of the constructor:
```yaml
_target_: code.optimization.batch_ADAM.AdamBatchOptimization

model: ${model}
data: ${data}
lr: ${params.lr}
num_workers: ${device.num_workers}
batch_size: ${params.batch_size}
pin_memory: ${device.pin_memory}
```
Once again the device-specic configuration refers the `device` component, while hyper-parameters point to the components
of `params`.

The `optimization.model` and `optimization.data` components point to `model` and `data` global keys respectively, as defined in the
[data](#data) and [model](#models) sections.

## Callbacks
Each callback in `callbacks` must be an instance of a [Pytorch Lighning callback](https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html?highlight=Callbacks#callbacks).
Callbacks are mainly used for checkpointing or logging.

Here we include the implementation of a [customized callback for evaluation](code/callbacks/evaluation_callbacks.py) 
that calls a specified evaluation metric any pre-definite amount of time (`evaluate_every`). This quantity can be specified
in model `iterations` or `epochs`, or in `seconds`,`minutes`,`hours` or `days` for increased flexibility.
This structure allows us to completely separate training and evaluation code. Another advantage is that the same evaluation
metric can be used for different models and architectures.

### Evaluation
Each evaluation metric is defined as an object that implements an `evaluate(optimization_procedure)` parameter that 
receives the Pytorch Lightning Module defining the optimization procedure and returns a `LogEntry` object, which 
specifies type of the entry and its value. Each evaluation metric is designed to be logger-agnostic (and data-agnostic
when possible).

Here we report the code for the evaluation procedure that is responsible for sampling and logging pictures
for an image generative model (such as the VAE reported in the previous examples):
```python
class ImageSampleEvaluation(Evaluation):
    def __init__(self, n_pictures=10, **kwargs):
        self.n_pictures = n_pictures
        self.kwargs = kwargs

    def evaluate(self, optimization: pl.LightningModule) -> LogEntry:
        model = optimization.model

        assert isinstance(model, GenerativeModel)

        x_gen = model.sample([self.n_pictures], **self.kwargs).to('cpu')

        return LogEntry(
            data_type=IMAGE_ENTRY,                          #Type of the logged object, to be interpreted by the logger
            value=make_grid(x_gen, nrow=self.n_pictures)    # Value to log
        )
```
The corresponding configuration is reported in the [experiment definition example](#defining-new-experiments) 
defined in the previous section.

### Checkpoints
The current implementation makes use of the Pytorch Ligthning [Checkpoint Callback](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.model_checkpoint.html)
with a [slight adaptation](code/callbacks/checkpoints.py) for Weight and Bias.
Basic Checkpoint callbacks are added into the 'config/logging' configuration.

TODO: Cumulative log callbacks

## Loggers
Since, in the [original Pytorch Lightning implementation](https://pytorch-lightning.readthedocs.io/en/stable/common/loggers.html?highlight=loggers), 
the code for logging differs across different loggers, we implement an extension of [TensorBoard](code/loggers/tensorboard.py)
and [Weights a& Bias](code/loggers/wandb.py) loggers that exposes a unified interface `log( name, log_entry, global_timestap)`.
The different `log_entry.data_type` are handled differently by different loggers. Currently only `scalar`, `scalars` and
`image` are implemented, but the wraper can be easily extended for other data types.
Here we report the example for the Wandb Logger wrapper:
```python

class WandbLogger(loggers.WandbLogger):
  
    def log(self, name: str, log_entry: LogEntry, global_step: int = None) -> None:
        # single scalar
        if log_entry.data_type == SCALAR_ENTRY:
            self.experiment.log({name: log_entry.value, 'trainer/global_step': global_step})
        # multiple scalars
        elif log_entry.data_type == SCALARS_ENTRY:
            entry = {'%s/%s' % (name, sub_name): v for sub_name, v in log_entry.value.items()}
            entry['trainer/global_step'] = global_step
            self.experiment.log(entry)
        # Image
        elif log_entry.data_type == IMAGE_ENTRY:
            self.experiment.log(data={name: wandb.Image(log_entry.value)}, step=global_step)
            plt.close(log_entry.value)
        # You can add other data-types to the chain of elif
        else:
            raise Exception('Data type %s is not recognized by WandBLogWriter' % log_entry.data_type)
```

## Lightning Trainer
The model training is based on the Pytorch Lightning Trainer, therefore all the corresponding parameters can be accessed
and modified.
This can be done from the configuration files (such as in the `config/logging/wandb.yaml` or 
the `config/device/laptop.yaml` files)
```yaml
# @package _global_
trainer:
  checkpoint_callback: False    # Disable the default model checkpoints
```
Or by terminal when launching the train script
```bash
python train.py experiment=MNIST_VAE +trainer.max_epochs=10
```

## Device

The device-specific configuration is defined in a separate `.yaml` configuration file in `config/device`, this include
(but is not limited to) directories and hardware-specific options.
The `device` configuration will be assigned depending on the environment variable 'DEVICE_NAME'.
As an example, if `DEVICE_NAME` is set to `laptop`, the configuration in `config/laptop.yaml` will be used.

This design allows us to define multiple devices (for deployment, training, testing) that are dynamically selected
based on the local value of `DEVICE_NAME`. Adding a new configuration is as easy as creating a new `.yaml` file to the
`config/device` folder and assigning the corresponding `DEVICE_NAME` on the device of interest.

Note that the trainer-specific configuration (such as number of gpus, tpus, accelerators, ...) can be specified direcly
from the device configuration using the following syntax:
```yaml
# @package _global_
trainer:
  gpus: 4
  ...
```
