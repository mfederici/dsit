import torch

from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

from datasets.cmnist_dist import make_joint_distribution


class CMNIST(MNIST):
    def __init__(self, root, version='CMNIST', sample_once=False, **params):
        super(CMNIST, self).__init__(root=root, **params, transform=ToTensor())

        assert version in ['CMNIST', 'd-CMNIST', 'y-CMNIST']

        self.dist = make_joint_distribution(version)
        self.sample_once = sample_once
        self.sampled_data = {}

    def __getitem__(self, index):
        x, d = super(CMNIST, self).__getitem__(index)

        sample = self.dist.condition_on('d', d).sample()

        # Concatenate an empty channel (red)
        x = torch.cat([x, x * 0], 0)

        # If the color is 1, make the empty channel the first (green)
        if sample['c'] == 1:
            x = torch.roll(x, 1, 0)

        return {'x': x, 'y': sample['y'], 'e': sample['e']}
