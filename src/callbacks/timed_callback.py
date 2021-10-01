import pytorch_lightning as pl
import time
from src.utils.time import TimeInterval


class TimedCallback(pl.Callback):
    def __init__(self, time):
        self.timer = TimeInterval(time)
        self._seconds = 0
        self.last_time = 0

    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.start_timer()

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
