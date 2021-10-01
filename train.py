import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig

from pytorch_lightning.loggers import WandbLogger

from framework.callbacks import EvaluationCallback
from framework.utils.wandb_utils import add_config
from pytorch_lightning import seed_everything


@hydra.main(config_path='config', config_name='config.yaml')
def parse(conf: DictConfig):
    if 'seed' in conf:
        seed_everything(conf.seed, workers=True)

    # Instantiate the Pytorch Lightning Trainer
    trainer = instantiate(conf.trainer)

    # On the first thread
    if trainer.global_rank == 0:
        # If using the Weights and Bias Logger
        if isinstance(trainer.logger, WandbLogger):

            # Add the hydra configuration to the experiment using the Wandb API
            add_config(trainer.logger.experiment, conf)

        # Add all the callbacks to the trainer (for logging, checkpointing, ...)
        if not trainer.fast_dev_run:

            # Evaluation Callbacks
            if 'evaluation' in conf:
                for name, eval_params in conf.evaluation.items():

                    # Instantiate the evaluator
                    evaluator = instantiate(eval_params['evaluator'])

                    # Create a corresponding evaluation callback
                    callback = EvaluationCallback(
                        name=name,
                        evaluator=evaluator,
                        evaluate_every=eval_params['evaluate_every']
                    )

                    # And add it to the trainer
                    trainer.callbacks.append(
                        callback
                    )

            # Add the rest of the callbacks
            for callback_conf in conf.callbacks:
                callback = instantiate(callback_conf)
                trainer.callbacks.append(callback)

    # Instantiate the optimization procedure, which is a Pytorch Lightning module
    optimization = instantiate(conf.optimization)

    # print the model structure
    print(optimization)

    # Call the Pytorch Lightning training loop
    trainer.fit(optimization)


if __name__ == '__main__':
    parse()
