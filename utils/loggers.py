import pytorch_lightning.loggers as loggers
import wandb
import matplotlib.pyplot as plt


class WandbLogger(loggers.WandbLogger):
    def log(self, name, value, type, global_step):
        if type == 'scalar':
            self.experiment.log({name: value}, step=global_step)
        elif type == 'scalars':
            entry = {'%s/%s' % (name, sub_name): v for sub_name, v in value.items()}
            self.experiment.log(entry, step=global_step)
        elif type == 'figure':
            self.experiment.log(data={name: wandb.Image(value)}, step=global_step)
            plt.close(value)
        else:
            raise Exception('Type %s is not recognized by WandBLogWriter' % type)


class TensorBoardLogger(loggers.TensorBoardLogger):
    def log(self, name, value, type, global_step):
        if type == 'scalar':
            self.log_metrics({name: value}, global_step=global_step)
        elif type == 'scalars':
            self.log_metrics({'%s/%s' % (name, sub_name): v for sub_name, v in value.items()}, global_step=global_step)
        elif type == 'figure':
            self.experiment.add_image(name, value, global_step=global_step)
            plt.close(value)
        else:
            raise Exception('Type %s is not recognized by TensorBoardLogger' % type)
