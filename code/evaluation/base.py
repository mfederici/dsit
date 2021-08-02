import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from code.loggers import LogEntry
from code.loggers.log_entry import SCALAR_ENTRY, SCALARS_ENTRY

TRAIN_STR = 'train'
VALID_STR = 'valid'
TEST_STR = 'test'
STRS = [TRAIN_STR, TEST_STR, VALID_STR]


class Evaluation:
    def evaluate(self, optimization: pl.LightningModule) -> LogEntry:
        raise NotImplemented()


class DatasetEvaluation(Evaluation):
    def __init__(self, evaluate_on=VALID_STR, n_samples=2048, batch_size=256, shuffle=False):
        self.data_loader = None
        self.evaluate_on = evaluate_on
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        assert evaluate_on in STRS

    def evaluate_batch(self, data, model):
        raise NotImplemented()

    def evaluate(self, optimization: pl.LightningModule) -> LogEntry:
        if self.data_loader is None:
            num_workers = optimization.train_dataloader().num_workers
            self.data_loader = DataLoader(optimization.data[self.evaluate_on],
                                          shuffle=self.shuffle,
                                          batch_size=self.batch_size,
                                          num_workers=num_workers)

        values = {}
        model = optimization.model
        evaluations = 0.
        device = next(model.parameters()).device

        model.eval()
        with torch.no_grad():
            for data in self.data_loader:
                if isinstance(data, dict):
                    for key in data:
                        data[key] = data[key].to(device)
                        if hasattr(data[key], 'shape'):
                            batch_len = data[key].shape[0]
                elif isinstance(data, list) or isinstance(data, tuple):
                    data_ = []
                    for d in data:
                        data_.append(d.to(device))
                        if hasattr(d, 'shape'):
                            batch_len = d.shape[0]
                    if isinstance(data, tuple):
                        data = tuple(data_)
                    else:
                        data = data_

                new_values = self.evaluate_batch(data, model)
                for k, v in new_values.items():
                    if k in values:
                        values[k] += v * batch_len
                    else:
                        values[k] = v * batch_len

                evaluations += batch_len
                if evaluations >= self.n_samples:
                    break

        values = {k: v/evaluations for k, v in values.items()}
        if len(values) == 1:
            for k in values:
                value = values[k]
            return LogEntry(
                data_type=SCALAR_ENTRY,
                value=value
            )
        else:
            return LogEntry(
                data_type=SCALARS_ENTRY,
                value=values
            )