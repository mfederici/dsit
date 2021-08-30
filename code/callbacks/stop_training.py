import pytorch_lightning as pl
import time
from code.utils.time import TimeInterval


class TrainForCallback(pl.Callback):
    def __init__(self, train_for):
        self.timer = TimeInterval(train_for)
        self._seconds = 0
        self.last_time = 0
        self.iterations = 0
        self.epochs = 0
        self.timer_active = False

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.start_timer()

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
        for name, value in pl_module.counters.items():
            setattr(self, name, value)

        self.global_step = trainer.global_step

        if self.timer.is_time(self):
            trainer.should_stop = True
