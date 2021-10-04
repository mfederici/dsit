import pytorch_lightning.loggers as loggers
import matplotlib.pyplot as plt
import wandb

from framework.logging.log_entry import LogEntry, IMAGE_ENTRY, SCALARS_ENTRY, SCALAR_ENTRY, PLOT_ENTRY


class WandbLogger(loggers.WandbLogger):
    def log(self, name: str, log_entry: LogEntry, global_step: int = None, counters: dict = None) -> None:
        if counters is None:
            entry = {}
        else:
            entry = {k: v for k, v in counters.items()}
        entry['trainer/global_step'] = global_step
        if log_entry.data_type == SCALAR_ENTRY:
            entry[name] = log_entry.value
            self.experiment.log(entry, commit=False)
        elif log_entry.data_type == SCALARS_ENTRY:
            for sub_name, v in log_entry.value.items():
                entry['%s/%s' % (name, sub_name)] = v
            self.experiment.log(entry, commit=False)
        elif log_entry.data_type == IMAGE_ENTRY:
            entry[name] = wandb.Image(log_entry.value)
            self.experiment.log(data=entry, step=global_step, commit=False)
            plt.close(log_entry.value)
        elif log_entry.data_type == PLOT_ENTRY:
            entry[name] = log_entry.value
            self.experiment.log(data=entry, step=global_step, commit=False)
            plt.close(log_entry.value)
        else:
            raise Exception('Data type %s is not recognized by WandBLogWriter' % log_entry.data_type)

