from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, Subset

MNIST_TRAIN_EXAMPLES = 50000


class MNISTWrapper(Dataset):
    def __init__(self, root, split, **params):
        assert split in ['train', 'valid', 'train+valid', 'test']

        dataset = MNIST(
            root=root,
            train=split in ['train', 'valid', 'train+valid'],
            transform=ToTensor(), **params)

        if split == 'train':
            dataset = Subset(dataset, range(MNIST_TRAIN_EXAMPLES))
        elif split == 'valid':
            dataset = Subset(dataset, range(MNIST_TRAIN_EXAMPLES, len(dataset)))
        elif not (split == 'test') and not (split == 'train+valid'):
            raise Exception('The possible splits are "train", "valid", "train+valid", "test"')

        self.dataset = dataset

    def __getitem__(self, item):
        x, y = self.dataset[item]
        return {'x': x, 'y': y}

    def __len__(self):
        return len(self.dataset)
