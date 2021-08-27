# Distribution Shift: An Information-Theoretic Analysis
This repository contains the code used for the results reported in the paper "Distribution Shift:An Information-Theoretic Analysis".
The code includes the implementation of mutual information computation and optimization for discrete distributions defined through tensors.

## Requirements
The following libraries are required to run the code:
- torch >= 1.0
- seaborn
- pandas
- tqdm

## Use
The notebook 'visualizations.ipynb' contains detailed descriptions of the experiments ad usage for one of the CMNIST distributions considered in this work.

The CMNIST variants used for the experiments reported in this work are defined in the '/datasets' folder. 'dataset.ipynb' shows an example of usage of the dataset using the default pytorch pipeline and 'Dataset' objects.
