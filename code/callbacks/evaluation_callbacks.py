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
        self.iterations = 0
        self.epochs = 0
        self.evaluator = evaluator
        self.name = name
        self.timer_active = False
        self.log_end = log_end
        self.log_beginning = log_beginning
        self.pause_timers = pause_timers

    def evaluate(self, pl_module: pl.LightningModule, trainer: pl.Trainer):
        if self.pause_timers:
            # Pause all the timers
            for callback in trainer.callbacks:
                if isinstance(callback, EvaluationCallback):
                    callback.stop_timer()

        log_entry = self.evaluator.evaluate(pl_module)
        trainer.logger.log(name=self.name, global_step=trainer.global_step, log_entry=log_entry)

        if self.pause_timers:
            # Restart the timers
            for callback in trainer.callbacks:
                if isinstance(callback, EvaluationCallback):
                    callback.start_timer()

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.start_timer()
        if self.log_beginning:
            self.evaluate(pl_module, trainer)

    def start_timer(self):
        self.last_time = time.time()
        self.timer_active = True

    def stop_timer(self):
        self.seconds
        self.timer_active = False

    def __getattribute__(self, item):
        if item == 'seconds':
            if self.timer_active:
                current_time = time.time()
                self._seconds += current_time - self.last_time
                self.last_time = current_time
            return self._seconds
        elif item == 'minutes':
            return self.seconds / 60.
        elif item == 'hours':
            return self.minutes / 60.
        elif item == 'days':
            return self.hours / 24.
        else:
            return super().__getattribute__(item)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.iterations = trainer.global_step
        self.epochs = pl_module.current_epoch

        if self.timer.is_time(self):
            self.timer.update(self)
            self.evaluate(pl_module, trainer)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.log_end:
            self.evaluate(pl_module, trainer)


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

    def evaluate(self, pl_module: pl.LightningModule, trainer: pl.Trainer):
        if self.mode == 'mean':
            entry = {name: np.mean(value) for name, value in self.outputs.items()}
        elif self.mode == 'last':
            entry = {name: value[-1] for name, value in self.outputs.items()}
        else:
            raise NotImplemented()

        log_entry = LogEntry(value=entry, data_type=SCALARS_ENTRY)
        trainer.logger.log(name=self.name, log_entry=log_entry, global_step=trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.iterations = trainer.global_step
        self.epochs = pl_module.current_epoch

        for name, value in outputs.items():
            if not name in self.outputs:
                self.outputs[name] = []
            self.outputs[name].append(value)

        if self.timer.is_time(self):
            self.timer.update(self)
            self.evaluate(pl_module, trainer)
