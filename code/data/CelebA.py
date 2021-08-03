from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor, CenterCrop, Compose
from torch.utils.data import Dataset


class CelebAWrapper(Dataset):
    def __init__(self, root, split, **params):
        assert split in ['train', 'valid', 'test']

        dataset = CelebA(
            root=root,
            split=split,
            transform=Compose([CenterCrop(218),ToTensor()]),
            **params)

        self.dataset = dataset

    def __getitem__(self, item):
        data = self.dataset[item]
        if isinstance(data, tuple):
            x, y = data
            return {'x': x, 'y': y}

        else:
            x = data
            return {'x': x}

    def __len__(self):
        return len(self.dataset)

