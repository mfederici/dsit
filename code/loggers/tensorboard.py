import pytorch_lightning.loggers as loggers
import matplotlib.pyplot as plt

from code.loggers.log_entry import SCALAR_ENTRY, SCALARS_ENTRY, IMAGE_ENTRY, LogEntry


class TensorBoardLogger(loggers.TensorBoardLogger):
    def log(self, name: str, log_entry: LogEntry, global_step=None):
        if log_entry.data_type == SCALAR_ENTRY:
            self.log_metrics({name: log_entry.value}, global_step=global_step)
        elif log_entry.data_type == SCALARS_ENTRY:
            self.log_metrics({'%s/%s' % (name, sub_name): v for sub_name, v in log_entry.value.items()},
                             global_step=global_step)
        elif log_entry.data_type == IMAGE_ENTRY:
            self.experiment.add_image(name, log_entry.value, global_step=global_step)
            plt.close(log_entry.value)
        else:
            raise Exception('Data type %s is not recognized by TensorBoardLogger' % log_entry.data_type)