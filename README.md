# Distribution Shift: An Information-Theoretic Analysis
This repository contains the code used for the results reported in the paper 
[Distribution Shift: An Information-Theoretic Analysis](https://arxiv.org/abs/2106.03783).

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

## Usage

### Setup
The required can be installed using [conda](https://www.anaconda.com/):
```shell
conda env create -f environment.yml
```
Secondly, the environment can be activate with:
```shell
conda activate dl-kit
```
In order to use the Weights & Bias logging run:
```shell
wandb init
```
This operation is optional if the desired logging option is TensorBoard, which can be enabled
using the flag `logging=tensorboard` when running the training script.

Lastly set a device name by defining the `DEVICE_NAME` environment variable.
```shell
export DEVICE_NAME=<A_VALID_DEVICE_NAME>
```
This will enable the corresponding configuration (in [config/device](config/device)).
The files [`laptop.yaml`](config/device/laptop.yaml) and [`workstation.yaml`](config/device/workstation.yaml) contain 
two examples of deployment configurations containing:
- dataset and experiments paths
- hardware specific configuration (number of GPUs and CPU cores).

### Model Training
Experiments can now run using:
```shell
python train.py <FLAGS>
```
Where `<FLAGS>` refers to any configuration flag define in `config` and handled by hydra.
If no experiment is specified, by default, the script will train a VIB model on the CMNIST dataset.

Other models can be trained using the `model` flag, while datasets can be changed using the `data` flag.
For example, to run the CDANN model on y-CMNIST, one can use:
```shell
python train.py model=CDANN data=y-CMNIST
```

Other flags allow to change optimization parameters, logging, evaluation and regularization schedule:
```shell
python train.py model=CDANN data=y-CMNIST params.lr=1e-3 params.n_adversarial_steps=20 train_for="2 hours"
```

Note that the `train_for` flag allows to specify the training duration in `iterations`, `epochs` or even `seconds`, `minutes`, `hours`.

### Notebooks
The [`error_decomposition.ipynb`](error_decomposition.ipynb) contains a detailed explanation regarding how test error
can be de-composed into **test information loss** and **latent test error**. This notebook
also includes details regarding the training procedure for discrete models and how the results reported in the paper
have been obtained.

The [`dataset.ipynb`](dataset.ipynb) contains a simple example of usage for the `CMNIST`, `d-CMNIST` and
`y-CMNIST` datasets to promote reproducibility and re-use of data-generating processes analyzed in this work.

