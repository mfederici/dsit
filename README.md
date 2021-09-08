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
- Get all the perks of a Pytorch [Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)
- Use the Weights & Bias [sweep tool](https://docs.wandb.ai/guides/sweeps) 
to easily define hyper-parameters search and launch multiple agents across different devices.
- Easily handle complex configuration thanks to the powerful [Hydra configuration management](https://hydra.cc/docs/intro/).
- No need to write any training/configuration script.

As a general concept, the framework is designed to that each piece can be maximally re-used:
the same model can be used with different architectures and dataset, the same architectures and evaluation procedures 
can be used with different models, and the same optimization procedure can be used for different models without the need to re-write any code.




# Installing
## Conda environment
Create a new `dl-kit` environment using conda:
```shell
conda env create -f environment.yml
```
and activate it:
```shell
conda activate dl-kit
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
# Example of the content of config/device/ivi-cluster.yaml to run experiment on a SLURM cluster

  gpus: 1                         # Specify the number of GPUs to use for the lightning trainer
  data_root: /ssdstore/datasets   # Root dataset directory
  experiments_root: /hddstore     # Root experiment directory
  download_files: False           # Flag to disable dataset download (from code)
  num_workers: 16                 # Number of workers used for data-loading
  pin_memory: True                # See pin_memory flag for the pytorch DataLoader
```
With this setup, the same code can be used on different machines since all the hardware-dependent configuration 
is grouped into the device `.yaml` configuration file. Further details can be found in the [device section](#device)

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
  name: ELBO/Validation         # Metric logged and defined in config/experient/MNIST_VAE.yaml
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

The results of the corresponding sweep can be visualized in the 
[Weights & Biases board](https://wandb.ai/mfederici/mnist_examples/sweeps/m0y9l5op)
![](https://user-images.githubusercontent.com/6851861/130940157-756357b8-e66d-456b-9a4f-17721a4fc4ec.png)

## Running on a SLURM cluster

The repository contains a few examples of [SLURM sbatch scripts](https://slurm.schedmd.com/sbatch.html) that can
be used to run experiments or sweeps on a SLURM cluster (such as Das5 or the ivi-cluster).

The [run_sweep_2h file](scripts/run_sweep_2h.sbatch) report the example used to run a sweep for a VAE model on MNIST 
once the corresponding wandb sweep has been created.

It is possible to start an agent on the das5 cluster using the provided scripts with:
```shell
sbatch --export=DEVICE_NAME=$DEVICE_NAME,SWEEP=<SWEEP_ID> scripts/run_sweep_2h.sbatch
```
in which the `--export` flag is used to pass variables to the script, and <SWEEP_ID> refer to the string
produced when running the `wandb sweep` command (see previous section).




# The run configuration
The configuration for each run is composed by the following main components:
- [**data**](#data): the data used for training the models. See section for further 
  information.
- [**model**](#Models) : the model to train. Each model must implement the logic regarding the loss computation
  (e.g. `VAE`, `GAN`, `VIB`, ...) and functionalities (e.g. `sample`, `reconstruct`, `classify`,...) in architecture and 
  data-agnostic fashion. Each model is an instance of a [Pytorch Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).
- [**optimization**](#optimization-procedures): the procedure used for optimizing the model. Definition on how the model is updated by the optimizer 
  (e.g. standanrd step update, adversarial training, joint training of two models, optimizer type, batch-creation procedure). Each optimization procedure is an 
  instance of a [Lighning Module](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html).

- [**params**](#hyper-parameters): collection of the model, architecture, optimization and data hyper-parameters (e.g. 
  number of layers, learning rate, batch size, regularization strength, ...). This design allows for easy definition of 
  [sweeps and hyper-parameter tuning](https://docs.wandb.ai/guides/sweeps).
- [**evaluation**](#evaluation): Dictionary containing metrics that need to be logged and their corresponding logging
  frequency.
- [**device**](#device): Definition of the hardware-specific parameters (such as paths, CPU cores, number of GPUs)
- [**trainer**](#trainer): Extra parameters passed to the [Lightning Trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)
- [**callbacks**](#callbacks): the callbacks called during training. Different callbacks can be used for logging, [model checkpointing](#checkpoints) or early stopping. 
  See [the corresponding documentation](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html)
  for further details. Note that callbacks are fully optional.
- [**logger**](#loggers): Definition of the logging procedure. Both TensorBoard and Weights & Bias are supported.
- [**run**](#run-details): Properties of the run such as name and the corresponding project


  While `data`, `model`, `architectures`, `optimization procedure`, `parameters`, `evaluation`, `callbacks` and `run` are experiment-specific, 
`device`, `logging` and `trainer` define global properties of the device on which the experiments are running, 
the logging and training parameters respectively.

## Defining new Experiments
Each experiment `.yaml` configuration file contains a definition of data, model, architectures, optimization procedure, 
hyper-parameters and callbacks. 
Here we report an example for the 
[Variational Autoencoder Model trained on the MNIST dataset](config/experiment/MNIST_VAE.yaml).
```yaml
# @package _global_
defaults:
  - /data: MNIST                  # On the MNIST dataset
  - /optimization: batch_ADAM     # Use batch-based optimization with Adam optimizer
  - /model: VAE                   # For a VAE model
  - /architectures: MNIST         # With the appropriate architectures
  - /evaluation:                  # And evaluate:
    - reconstruction              # The reconstructions (on validation set)
    - generation                  # The images sampled from the model
    - elbo                        # The value of ELBO (on train and validation)

seed: 42                          # Set a fixed seed for reproducibility

# Name of the run and project
run:
  project: VAE_experiments        # set the name of the project (`noname` by default)
  name: My_First_VAE              # Name of the run (used by Weights & Biases)
  
# Values of the hyper-parameters
params:
  z_dim: 64                       # Number of latent dimensions
  beta: 0.5                       # Value of regularization strength
  lr: 1e-3                        # Learning rate
  batch_size: 128                 # Batch size
  encoder_layers: [ 1024, 128 ]   # Layers of the encoder model
  decoder_layers: [ 128, 1024 ]   # Layers of the decoder model

# Parameters for the evaluation (such as frequency or number of samples)
eval_params:
  elbo:
    every: 1 minute               # Compute ELBO every minute 
    n_samples: 2048               # using 2048 samples
  samples:
    every: 1 minute               # Visualize samples for the generative model every minute
    n_samples: 10                 # Number of visualized samples
  reconstruction:
    every: 1 minute               # Show the reconstruction every minutes
    evaluate_on: valid            # of pictures from the validation set
    n_samples: 10                 # pick 10 of them
```
The `# @package _global_` line is used to indicate that the following keys are global, while `defaults` specifies 
the values for `data`, `optimization`, `model`, `architectures` and `evluation` procedures respectively. 
In other words, the dictionary defined in the files [`config/data/MNIST.yaml`](config/data/MNIST.yaml), 
[`config/optimization/batch_ADAM.yaml`](config/optimization/batch_ADAM.yaml), [`config/model/VAE.yaml`](config/model/VAE.yaml), 
[`config/architectures/MNIST.yaml`](`config/architectures/MNIST.yaml`)  and 
[`config/evaluation/reconstruction.yaml`](config/evaluation/reconstruction.yaml) (and the other evaluation metrics) are added 
to the respective keys.
All the configuration that is used to launch an experiment will be stored inside a `hydra` folder for the 
corresponding run.

Further information regarding configuration packages and overrides can be found 
[here](https://hydra.cc/docs/advanced/overriding_packages).

All the hyper-parameters regarding architectures and optimization are grouped under the `params` dictionary.
This structure makes the configuration file extremely easy to read and understand, with the added value of making it easy
to change the value of hyper-parameters.

The parameters required by the evaluation metrics are grouped in the `eval_params` dictionary,
which is responsible to define details regarding the evaluation procedure (and the logging frequency).

To summarize, each experiments defines the following:
- `data`: data configuration
- `architectures`: which set of architectures to use
- `model`: which model (objective, what architectures and hyperparameters are needed)
- `optimization`: definition of the optimization procedure (SGD, Adversarial, others...)
- `params`: list of hyper-parameters (from model, architectures, data and optimization procedure)
- `evaluation` (optional): list of evaluation metrics to log
- `eval_params` (optional): list of evaluation parameters

Note that the values defined in the experiments can be altered directly from the command line.
For example, by running
```shell
python train.py +experiment=MNIST_VAE params.beta=0.0001 train_for="30 minutes"
```
one can overwrite the value of `params.beta`, changing it from `0.01` to `0.0001`, and set the
total training time to `30 minutes`.

Further details regarding the aforementioned configuration components can be found in the following sections.



# Creating new Implementations

Adding new models, datasets and architectures to the frameworks requires implementing the code and creating the 
corresponding configuration files.
Here we report the conventions used to define the different components: 

## Data
The datasets definition are collected in the [`data` configuration folder](config/data). Each data object consist
of a dictionary specifying the parameters for the different splits. In this example, we consider `train`, `valid`, and `test`
for training, validation and testing purpose respectively, but different keys can be added to the `data` dictionary if necessary:
```yaml
# Content of /config/data/MNIST.yaml. the corresponding keys are added under `data`
train:                                      # Definition of the training set
  _target_: code.data.MNIST.MNISTWrapper    # Class 
  root: ${device.data_root}                 # Initialization parameters (passed to the constructor)
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
While reading the configuration, the variables `${NAME.ATTRIBUTE}` will be resolved 
to the corresponding value.
All the device-dependent parameters (such as the data directory and the flag to enable downloading)
refer to the `${device}` variable. This strategy allows to easily deploy the same model to different devices. Further details can
be found in the [device section](#device).

TorchVision, TorchAudio or other existing datasets class definitions can be referenced directly by specifying
the appropriate `_target_` (e.g. `_target_: torchvision.datasets.MNIST` for the default torchvision MNIST dataset).

The instantiated `data` dictionary is passed to the constructor of the [optimization procedure](#optimization-procedures),
in which the data-loaders are defined.

## Models
The model configuration defines the parameters and references to required architectures. The model code is designed to be completely data-agnostic
so that the same logic can be used across different experiments without any re-writing or patch.
Here we report the example code for the `VariationalAutoencoder` [model](code/models/unsupervised/VAE.py):
```python
# the VariationalAutoencoder class is a torch.nn.Module that: 
#  - inherits the `sample()` method from GenerativeModel,
#  - inherits the `encode(x)` method from RepresentationLearningModel.
# This inheritance structure is useful to better define what models can be used for and how
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
        :param encoder: the encoder architecture (as a nn.Module : (Tensor) -> Distribution)
        :param decoder: the decoder architecture (as a nn.Module : (Tensor) -> Distribution)
        :param prior: architecture representing the prior (as a nn.Module : () -> Distribution)
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

Note that the code for the Variational Autoencoder model is completely independent of the specific dataset or
architectures. Therefore, the same model can be used on all datasets without any need to rewrite or change the code as 
long as the different data types are handled correctly by the respective encoder and decoder architectures.

The corresponding [configuration file](config/model/VAE.yaml) defines where the required architectures and parameters 
are defined:
```yaml
_target_: code.models.unsupervised.VariationalAutoencoder   # The VAE class defined above
prior: ${architectures.prior}                               # pass the architecture.prior as the prior
decoder: ${architectures.decoder}                           # architectures.decoder as the decoder
encoder: ${architectures.encoder}                           # archidectures.encoder as the encoder
beta: ${params.beta}                                        # and params.beta for the regularization strength

```
Note that this configuration acts as linker that defines where the different components have to be looked up in other 
parts of the configuration. This is useful because one can use the VAE model on completely different datasets
just by swapping in and out different values for the `architectures` without modifying the code defining the logic
of the model.

## Architectures 
Different architectures are implemented in the `code/architectures` folder. Each architecture is designed for a specific 
role (e.g. `Encoder`, `Decoder`, `Predictor`, ...), as a result the same architecture can be used in multiple models.
Since the architecture code is data-dependent, each dataset will correspond to a different set of architectures.

Here we report the example for the `VariationalAutoencoder` `Encoder` on `MNIST`:
```python
INPUT_SHAPE = [1, 28, 28]
N_INPUTS = 28*28
N_LABELS = 10


# Model for q(Z|X)
# The forward method is designed to map from a Tensor to a Distribution
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
The [corresponding configuration](config/architectures/MNIST.yaml) links the parameters of the encoder model to the 
`params` values:
```yaml
# @package _global_
architectures:
  encoder:
    _target_: code.architectures.MNIST.Encoder  # Class to instantiate
    layers: ${params.encoder_layers}            # List passed as the layers definition
    z_dim: ${params.z_dim}                      # Size of latent dimensions

# The rest of the architectures are defined below...
```
Note that the `architecture` file defines the linking for the architectures used by any model (and not only the `VAE`).
At run time only the required keys will be dynamically looked up.

## Optimization procedures
The optimization procedure is designed to contain the logic regarding how the model is updated over time.
This includes the definition of optimizers, data-loaders (or environments for reinforcement learning) and learning-rate schedulers.
Once again, the optimization procedure is designed to be modular and as independent from the other components as 
possible (model-agnostic). Here we report the example
for the a [batch-based training procedure with the ADAM optimizer](code/optimization/batch_ADAM.py):
```python
# Each optimization procedure is a pytorch lightning module
class AdamBatchOptimization(Optimization):
    def __init__(self,
                 model: Model,          # The model to optimize (as a nn.Module)
                 data: dict,            # The dictionary of Datasets defined in the previous 'Data' section
                 num_workers: int,      # Number of workers for the data_loader
                 batch_size: int,       # Batch size
                 lr: float,             # Learning rate
                 pin_memory: bool=True  # Flag to enable memory pinning
                 ):
        super(AdamBatchOptimization, self).__init__()
        
        # Assigned the variables passed to the constructor as internal attributes
        self.model = model
        self.data = data

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.pin_memory = pin_memory

    # Overrides the pl.LightningModule train_dataloader which is used by the Trainer
    def train_dataloader(self) -> DataLoader:
        # Return a DataLoader for the `train` dataset with the specified batch_size
        return DataLoader(self.data['train'],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=self.pin_memory)

    # The training step simply returns the computation from the model 
    # and logs the loss entries
    def training_step(self, data, data_idx) -> STEP_OUTPUT:
        # Compute the loss using the compute_loss function from the model
        loss_items = self.model.compute_loss(data, data_idx)
        
        # Log the loss components
        for name, value in loss_items.items():
            self.log('Train/%s' % name, value)
            
        # Increment the iteration counts.
        # The self.counters dictionary can be used to define custom counts
        # (e.g number of adversarial/generator iterations during adversarial training)
        self.counters['iteration'] += 1
        
        return loss_items
    
    # Instantiate the Adam optimizer passing the model trainable parameters
    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)
```
Each optimization procedure is a Pytorch Lightning module, therefore it is possible to extend all the corresponding
 [functions](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html) for customized 
training/data loading.

The corresponding [configuration file](config/optimization/batch_ADAM.yaml) simply defines the references for the
parameters of the constructor accessing either the hyper-parameters in `params` or device-specific configuration in 
`device` (which depends on the number of CPU cores and presence of a GPU):
```yaml
_target_: code.optimization.batch_ADAM.AdamBatchOptimization

model: ${model}                         # Reference to the model
data: ${data}                           # And dataset(s)
lr: ${params.lr}                        # Pointing to the value of learning rate
batch_size: ${params.batch_size}        # and batch size in `params`
num_workers: ${device.num_workers}      # Looking up the number of workers
pin_memory: ${device.pin_memory}        # and if memory pinning is used from the device config
```

The `optimization.model` and `optimization.data` components point to `model` and `data` global keys respectively, which 
are defined in the [data](#data) and [model](#models) sections.


## Evaluation
For easy and flexible evaluation, we defined a customized and extensible evaluation procedure using 
[Pytorch Lighning callbacks](https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html?highlight=Callbacks#callbacks).
A custom [`EvaluationCallback`](code/callbacks/evaluation_callbacks.py) calls an 
[`EvaluationProcedure`](code/evaluation/base.py) at pre-defined time intervals (in `global_steps`, `epochs`, `seconds`, 
`minutes`, `hours`, `days` or any unit defined in the `counter` dictionary attribute of the `OptimizationProcedure`).

This allows to easily define `Evaluation` procedures that map an instance of `Optimization` (containing `model` and `data`)
to a `LogEntry`. Here we report the example for a class creating images samples for a generative model:
```python

# The evaluation must extend the base class Evaluation
class ImageSampleEvaluation(Evaluation):
    # Constructor containing the evaluation parameters.
    def __init__(self, n_pictures=10, sampling_params=None):
        self.n_pictures = n_pictures
        self.sampling_params = sampling_params if not(sampling_params is None) else dict()
    
    # Definition of the evaluation function 
    def evaluate(self, optimization: pl.LightningModule) -> LogEntry:
      
      # Get the model from the optimization procedure
      model = optimization.model
        
      # And evaluate the model 
      return self.evaluate_model(model)

    
    # Evaluation function for the model (which needs to be a GenerativeModel 
    # This means that it has a `sample()` function to generate samples
    def evaluate_model(self, model: GenerativeModel):
      
      # Generate the samples
      with torch.no_grad():
        x_gen = model.sample([self.n_pictures], **self.sampling_params).to('cpu')

      # Return a log-entry with the type IMAGE_ENTRY 
      return LogEntry(
      data_type=IMAGE_ENTRY,  # Type of the logged object, to be interpreted by the logger
      value=make_grid(x_gen, nrow=self.n_pictures)  # Value to log
      )
    
```
The corresponding [configuration file](config/evaluation/generation.yaml) defines the name that will be visualized in the Tensorboard/Weights & Bias log and the logging frequency as follows:
```yaml
Images/Samples:                                             # The key is used to define the entry name
  evaluate_every: ${eval_params.samples.every}              # Frequency of evaluation
  evaluator:                                                # Evaluator object as defined above
    _target_: code.evaluation.image.ImageSampleEvaluation   # Class for the evaluation
    n_pictures: ${eval_params.samples.n_samples}            # Parameters passed to __init__()
```

The parameters for the evaluation metrics are collected in the `eval_params` dictiornary, which is defined in the experiment file.
Note that the same experiment can have multiple evaluation metrics as long as the entry names (e.g. `Images/Samples` 
are different from each other).


## Callbacks
Each callback in `callbacks` must be an instance of a [Pytorch Lighning callback](https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html?highlight=Callbacks#callbacks).
Callbacks are mainly used for 
[checkpointing](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.model_checkpoint.html) or 
[early stopping](https://pytorch-lightning.readthedocs.io/en/latest/common/early_stopping.html).

Here we include an implementation of a [`TrainDurationCallback`](code/callbacks/stop_training.py) which allows one to define custom training time (in `iterations`,
`global_steps`, `epochs`, `seconds`, `minutes`, `hours`, `days` as for the evaluation metrics).
The `TrainDurationCallback` is added to the list of used callbacks by default and reads the train duration from a `train_for` variable
in the configuration.

Additional callbacks can be used for other custom functions that need to act on the optimization procedure, 
model or its components.

### Checkpoints
The current implementation makes use of the Pytorch Lightning [Checkpoint Callback](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.model_checkpoint.html)
with a [slight adaptation](code/callbacks/checkpoints.py) for Weight and Bias.
Basic Checkpoint callbacks are added into the 'config/logging' configuration.



## Loggers
Since, in the [original Pytorch Lightning implementation](https://pytorch-lightning.readthedocs.io/en/stable/common/loggers.html?highlight=loggers), 
the code for logging differs across different loggers, we implement an extension of [TensorBoard](code/loggers/tensorboard.py)
and [Weights a& Bias](code/loggers/wandb.py) loggers that exposes a unified interface `log( name, log_entry, global_timestap)`.
The different `log_entry.data_type` are handled differently by different loggers. Currently only `scalar`, `scalars` and
`image` are implemented, but the wraper can be easily extended for other data types.
Here we report the example for the Wandb Logger wrapper:
```python

class WandbLogger(loggers.WandbLogger):
    # Custom log function designed to handle typed log-entries.
    # if defined, any additional counter in the optimization procedure is also logged.
    def log(self, name: str, log_entry: LogEntry, global_step: int = None, counters: dict = None) -> None:
        if counters is None:
            entry = {}
        else:
            entry = {k: v for k, v in counters.items()}
        entry['trainer/global_step'] = global_step
        if log_entry.data_type == SCALAR_ENTRY:
            entry[name] = log_entry.value
            self.experiment.log(entry, commit=False)
        elif log_entry.data_type == SCALARS_ENTRY:
            for sub_name, v in log_entry.value.items():
                entry['%s/%s' % (name, sub_name)] = v
            self.experiment.log(entry, commit=False)
        elif log_entry.data_type == IMAGE_ENTRY:
            entry[name] = wandb.Image(log_entry.value)
            self.experiment.log(data=entry, step=global_step, commit=False)
            plt.close(log_entry.value)
        elif log_entry.data_type == PLOT_ENTRY:
            entry[name] = log_entry.value
            self.experiment.log(data=entry, step=global_step, commit=False)
            plt.close(log_entry.value)
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
Or by terminal when launching the train script (e.g. enabling a 
[fast dev run](https://pytorch-lightning.readthedocs.io/en/latest/common/debugging.html))
```bash
python train.py experiment=MNIST_VAE +trainer.fast_dev_run=True
```

## Device
The device-specific configuration is defined in a separate `.yaml` configuration file in `config/device`, this include
(but is not limited to) directories and hardware-specific options.
The `device` configuration will be assigned depending on the environment variable 'DEVICE_NAME'.
As an example, if `DEVICE_NAME` is set to `laptop`, the configuration in `config/laptop.yaml` will be used.

This design allows us to define multiple devices (for deployment, training, testing) that are dynamically selected
based on the local value of `DEVICE_NAME`. Adding a new configuration is as easy as creating a new `.yaml` file to the
`config/device` folder and assigning the corresponding `DEVICE_NAME` on the device of interest.

## Run Details
The run configuration object is used to define the name associated to the run (`run.name`) and the name of the 
corresponding project (`run.project`). These properties can be accessed and modified form the command line
or by specifying them in the experiment definition.

# Loading models
The [loading.ipynb](loading.ipynb) notebook reports an example on how models can be easily retrieved directly from the
Weights & Bias Api and the Hydra `instantiate` function.