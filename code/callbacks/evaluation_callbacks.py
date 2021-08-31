import pytorch_lightning as pl
import time
import numpy as np

from code.loggers.log_entry import SCALARS_ENTRY, LogEntry
from code.utils.time import TimeInterval


class EvaluationCallback(pl.Callback):
    # Only one unit is currently supported: `10 minutes 10 seconds` is not a valid value
    def __init__(self, name, evaluator, evaluate_every, log_end=True, log_beginning=True, pause_timers=True):
        self.timer = TimeInterval(evaluate_every)
        self._seconds = 0
        self.last_time = 0
        self.evaluator = evaluator
        self.name = name
        self.timer_active = False
        self.log_end = log_end
        self.log_beginning = log_beginning
        self.pause_timers = pause_timers

    def evaluate(self, pl_module: pl.LightningModule):
        log_entry = self.evaluator.evaluate(pl_module)
        if hasattr(pl_module, 'counters'):
            counters = pl_module.counters
        else:
            counters = None
        pl_module.trainer.logger.log(
            name=self.name,
            log_entry=log_entry,
            global_step=pl_module.trainer.global_step,
            counters=counters
        )

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.start_timer()
        if self.log_beginning:
            self.evaluate(pl_module)

    def start_timer(self):
        self.last_time = time.time()
        self.timer_active = True

    def stop_timer(self):
        self.get_seconds()
        self.timer_active = False

    def get_seconds(self):
        if self.timer_active:
            current_time = time.time()
            self._seconds += current_time - self.last_time
            self.last_time = current_time
        return self._seconds

    def get_counters(self, pl_module):
        if hasattr(pl_module, 'counters'):
            counters = {k: v for k, v in pl_module.counters.items()}
        else:
            counters = {}
        counters['global_step'] = pl_module.trainer.global_step
        counters['epoch'] = pl_module.current_epoch
        counters['second'] = self.get_seconds()
        counters['minute'] = counters['second'] // 60
        counters['hour'] = counters['minute'] // 60
        counters['day'] = counters['hour'] // 24

        return counters

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        for name, value in pl_module.counters.items():
            setattr(self, name, value)

        counters = self.get_counters(pl_module)

        if self.timer.is_time(counters):
            self.timer.update(counters)
            if self.pause_timers:
                # Pause all the timers
                for callback in trainer.callbacks:
                    if isinstance(callback, EvaluationCallback):
                        callback.stop_timer()

            self.evaluate(pl_module)

            if self.pause_timers:
                # Restart the timers
                for callback in trainer.callbacks:
                    if isinstance(callback, EvaluationCallback):
                        callback.start_timer()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.log_end:
            self.evaluate(pl_module)


class LossItemsLogCallback(EvaluationCallback):
    def __init__(self, log_every, mode='mean'):
        super(LossItemsLogCallback, self).__init__(pause_timers=False,
                                                   evaluate_every=log_every,
                                                   log_end=True,
                                                   log_beginning=False,
                                                   name='Training',
                                                   evaluator=None)
        self.outputs = {}
        self.mode = mode
        assert mode in ['mean', 'last']

    def evaluate(self, pl_module: pl.LightningModule):
        if self.mode == 'mean':
            entry = {name: np.mean(value) for name, value in self.outputs.items()}
        elif self.mode == 'last':
            entry = {name: value[-1] for name, value in self.outputs.items()}
        else:
            raise NotImplemented()

        log_entry = LogEntry(value=entry, data_type=SCALARS_ENTRY)
        if hasattr(pl_module, 'counters'):
            counters = pl_module.counters
        else:
            counters = None

        pl_module.trainer.logger.log(
            name=self.name,
            log_entry=log_entry,
            global_step=pl_module.trainer.global_step,
            counters=counters
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        counters = self.get_counters(pl_module)

        if self.timer.is_time(counters):
            self.timer.update(counters)
            self.evaluate(pl_module)
