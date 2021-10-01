from src.callbacks.timed_callback import TimedCallback


class TrainDurationCallback(TimedCallback):
    def __init__(self, train_for):
        super(TrainDurationCallback, self).__init__(time=train_for)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        counters = self.get_counters(pl_module)

        if self.timer.is_time(counters):
            trainer.should_stop = True
