import pytorch_lightning.loggers as loggers
import matplotlib.pyplot as plt
import wandb

from code.loggers import LogEntry
from code.loggers.log_entry import IMAGE_ENTRY, SCALARS_ENTRY, SCALAR_ENTRY


class WandbLogger(loggers.WandbLogger):
    def log(self, name: str, log_entry: LogEntry, global_step: int = None) -> None:
        if log_entry.data_type == SCALAR_ENTRY:
            self.experiment.log({name: log_entry.value, 'trainer/global_step': global_step})
        elif log_entry.data_type == SCALARS_ENTRY:
            entry = {'%s/%s' % (name, sub_name): v for sub_name, v in log_entry.value.items()}
            entry['trainer/global_step'] = global_step
            self.experiment.log(entry)
        elif log_entry.data_type == IMAGE_ENTRY:
            self.experiment.log(data={name: wandb.Image(log_entry.value)}, step=global_step)
            plt.close(log_entry.value)
        else:
            raise Exception('Data type %s is not recognized by WandBLogWriter' % log_entry.data_type)

