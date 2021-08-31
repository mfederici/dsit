import torch
import torchvision

from torch.utils.data import Dataset, Subset
from torchvision.transforms import ToTensor

from code.data.cmnist_dist import make_joint_distribution, CMNIST_NAME, CMNIST_VERSIONS

CMNIST_SIZE = 28 ** 2 * 2
CMNIST_SHAPE = [2, 28, 28]
CMNIST_N_CLASSES = 2
CMNIST_N_ENVS = 2

MNIST_TRAIN = 'train'
MNIST_VALID = 'valid'
MNIST_TEST = 'test'
MNIST_TRAIN_VALID = 'train+valid'
MNIST_TRAIN_SPLITS = [MNIST_TRAIN, MNIST_VALID, MNIST_TRAIN_VALID]
MNIST_SPLITS = MNIST_TRAIN_SPLITS + [MNIST_TEST]
MNIST_TRAIN_EXAMPLES = 50000


# Wrapper for the torchvision MNIST dataset with validation split
class MNIST(Dataset):
    def __init__(self, root, split, **params):
        super(MNIST, self).__init__()

        dataset = torchvision.datasets.MNIST(root=root,
                                             train=split in MNIST_TRAIN_SPLITS,
                                             transform=ToTensor(), **params)

        if split == MNIST_TRAIN:
            dataset = Subset(dataset, range(MNIST_TRAIN_EXAMPLES))
        elif split == MNIST_VALID:
            dataset = Subset(dataset, range(MNIST_TRAIN_EXAMPLES, len(dataset)))
        elif not (split == MNIST_TEST) and not (split == MNIST_TRAIN_VALID):
            raise Exception('The possible splits are %s' % ', '.join(MNIST_SPLITS))
        self.dataset = dataset

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return {'x': x, 'y': torch.LongTensor([y])}

    def __len__(self):
        return len(self.dataset)


# Implementation of the CMNIST, d-CMNIST and y-CMNIST datasets for pytorch
class CMNIST(MNIST):
    def __init__(self, root, version=CMNIST_NAME, sample_once=False, t=1, **params):
        super(CMNIST, self).__init__(root=root, **params)

        assert version in CMNIST_VERSIONS
        assert t in [0, 1]

        self.dist = make_joint_distribution(version).condition_on('t', t)
        self.sample_once = sample_once
        self.sampled_data = {}

    def __getitem__(self, index):
        if index in self.sampled_data:
            data = self.sampled_data[index]
        else:
            data = super(CMNIST, self).__getitem__(index)
            x = data['x']
            d = data['y']

            # sample from p(e,y,c|d) to determine color, label and environment
            sample = self.dist.condition_on('d', d).sample()

            # Concatenate an empty channel (red)
            x = torch.cat([x, x * 0], 0)

            # If the color is 1, make the empty channel the first (green)
            if sample['c'] == 1:
                x = torch.roll(x, 1, 0)

            data = {'x': x, 'y': sample['y'], 'e': sample['e']}
        if self.sample_once:
            self.sampled_data[index] = data
        return data
