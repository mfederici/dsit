import torch.nn as nn


class Model(nn.Module):
    def compute_loss(self, data, data_idx):
        raise NotImplemented()