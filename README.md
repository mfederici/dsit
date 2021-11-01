# An Information-theoretic Approach to Distribution Shifts
<img width=90% alt="Distribution Shift example" src="https://user-images.githubusercontent.com/6851861/139700177-39c857e8-73ad-40ca-b42c-d80f3110c125.gif">

This repository contains the code used for the results reported in the paper 
[An Information-theoretic Approach to Distribution Shifts](https://arxiv.org/abs/2106.03783).

The code includes the implementation of:
- [Three variants](src/data/CMNIST.py) of the CMNIST dataset:
  - CMNIST
  - d-CMNIST
  - y-CMNIST
- Discrete Models
  - [Mutual information computation and optimization](src/discrete/distribution/distribution.py) 
  for discrete distributions defined through tensors.
  - [Discrete encoders](src/discrete/models) defined through a normalized probability mapping matrix. 
- Neural Network-based Models
  - [Variational Information Bottleneck](src/models/VIB.py) (VIB)
  - [Domain Adversarial Neural Networks](src/models/DANN.py) (DANN)
  - [Invariant Risk Minimization](src/models/IRM.py) (IRM)
  - [Variance-based Risk Extrapolation](src/models/VREx.py) (VREx)
  - [Conditional Domain Adversarial Neural Networks](src/models/DANN.py) (CDANN)

The implementations are based on the [Pytorch Lightning](https://www.pytorchlightning.ai/) framework, 
[hydra](https://hydra.cc/) is used for configuration management,
while the [wandb](https://wandb.ai/) library handles logging and hyper-parameter sweeps 
([TensorBoard](https://www.tensorflow.org/tensorboard) logging is also available).

Further information on the framework and design paradigms used in this implementation can be found 
[here](https://github.com/mfederici/dl-kit/tree/master).

## Requirements
### Conda Environment
The required packages can be installed using [conda](https://www.anaconda.com/):
```shell
conda env create -f environment.yml
```
Once the environment has been created, it can be activated with:
```shell
conda activate dl-kit
```

### Device configuration
Set a device name by defining the `DEVICE_NAME` environment variable.
```shell
export DEVICE_NAME=<A_VALID_DEVICE_NAME>
```
This will enable the corresponding configuration (in [config/device](config/device)).
The files [`laptop.yaml`](config/device/laptop.yaml) and [`workstation.yaml`](config/device/workstation.yaml) contain 
two examples of deployment configurations containing:
- dataset and experiments paths
- hardware specific configuration (number of GPUs and CPU cores).

It is possible to create new customized device configuration by
1) Creating a `my_device.yaml` configuration file containing:
```yaml
data_root: <path_to_dataset_root>             # Path
experiments_root: <path_to_experiment_root>   # Path
download_files: <enable_dataset_download>     # True/False
num_workers: <number_of_CPU_for_data_loading> # Integer
pin_memory: <enable_pin_memory>               # True/False
gpus: <number_of_GPUs>                        # Integer                             
auto_select_gpus: <enable_auto_GPU_selection> # True/False
```
2) Setting the `DEVICE_NAME` environment variable to `my_device`:
```shell
export DEVICE_NAME=my_device
```

### Weights & Bias logging (and Sweep)
In order to use the Weights & Bias logging run:
```shell
wandb init
```
This operation is optional if the desired logging option is TensorBoard, which can be enabled
using the flag `logging=tensorboard` when running the training script.


## Datasets

<img  width=70% alt="Graphical Models" src="https://user-images.githubusercontent.com/6851861/138869778-10e88f39-a371-43f7-b939-bf58bc201236.png">


The code contains the implementation of the three variations of the Colored MNIST dataset 
(`CMNIST`,`d-CMNIST`, `y-CMNIST`), and three corresponding versions used for validation and hyper-parameter search 
(`CMNIST_valid`,`d-CMNIST_valid`, `y-CMNIST_valid`). In the former versions, models are trained on the train+validation
sets and evaluated on the test set. In the validation settings, models are trained on the train set and evaluated on
the (disjoint) validation set.
<center>
<img alt="CMNIST_samples" src="https://user-images.githubusercontent.com/6851861/138869050-9a400b55-1d07-4727-b4f9-fc59fa0c1800.png">
</center>

The [`dataset.ipynb`](dataset.ipynb) contains a simple example of usage for the `CMNIST`, `d-CMNIST` and
`y-CMNIST` datasets. The current implementation is based on the PyTorch `Dataset` class to promote reproducibility and 
re-usability.


## Training

### Discrete models and direct criteria optimization
![Criteria](https://user-images.githubusercontent.com/6851861/138870256-0e1bfe78-9f01-484f-8961-f33faa7dde6b.png)
![Discrete trajectories](https://user-images.githubusercontent.com/6851861/138878641-8ee1f63b-7d43-4876-ab9e-9e40e91bcf3f.png)

The discrete models can be trained using the command
```shell
python train_discrete.py
```
which will produce a `.csv` (`results/discrete.csv` by default) containing the values of train and test cross-entropy for model
optimized following the Information Bottleneck, Independence, Sufficiency and Separation criteria
on the CMNIST, d-CMNIST and y-CMNIST datasets for different regularization strength.

Similarly to the neural-network models training, the hyper-parameters can be changed
either by editing the [`discrete_config/config.yaml`](discrete_config/config.yaml) file, or by 
specifying the corresponding flags when launching the training script.

The [`error_decomposition.ipynb`](error_decomposition.ipynb) contains a detailed explanation regarding how test error
can be de-composed into **test information loss** and **latent test error**. This notebook
also includes details regarding the training procedure for discrete models and how the results reported in the paper
have been obtained.


<img  width=50% alt="Error Decomposition" src="https://user-images.githubusercontent.com/6851861/138878026-111128b4-b672-420b-8c17-c06e832a3a05.png">


### Neural Network Models
![MLP results](https://user-images.githubusercontent.com/6851861/138880275-e6b247b7-7228-4f63-908a-3db16fe4ad83.png)

Each model can be trained using the `train.py` script using the following command
```shell
python train.py <FLAGS>
```
Where `<FLAGS>` refers to any configuration flag defined in `config` and handled by hydra.
If no experiment is specified, by default, the script will train a VIB model on the CMNIST dataset.
Other models can be trained specifying the `model` flag (`VIB`,`DANN`,`IRM`,`CDANN`,`VREx`), 
while datasets can be changed using the `data` flag (`CMNIST`,`d-CMNIST`, `y-CMNIST`). 


For example, to run the CDANN model on y-CMNIST, one can use:
```shell
python train.py model=CDANN data=y-CMNIST
```

Other flags allow to change optimization parameters, logging, evaluation and regularization schedule.
The command
```shell
python train.py model=CDANN data=y-CMNIST params.lr=1e-3 params.n_adversarial_steps=20 train_for="2 hours"
```
will train a CDANN model on y-CMNIST with learning rate 10^{-3}, using 20 trainin steps of the discriminator for each 
generator step with TensorBoard logging for a total training time of 2 hours.
Note that the `train_for` flag allows to specify the training duration in `iterations`, `epochs` or even `seconds`, `minutes`, `hours`.

#### Weights & Bias Sweeps
The hyper-parameters sweeps used to produce the plots in the paper can be found
in the [`sweeps`](sweeps) directory.

Running sweeps requires the [initialization of the Weights & Bias](#weights--bias-logging-and-sweep).
To run all the experiments used to produce the plot in Figure 3 (bottom row), one can use:
```shell
wandb sweep sweeps/sweep_MLP.yml
```
which will return a unique `<SWEEP_ID>`.

Each sweep agent can then be launched using:
```shell
wandb agent <SWEEP_ID>
```
from the project directory.

