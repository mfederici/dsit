import torch
import numpy as np
from torchvision.utils import make_grid
from code.evaluation import Evaluation
from code.loggers import LogEntry
import pytorch_lightning as pl

from code.loggers.log_entry import IMAGE_ENTRY


class ImageReconstructionEvaluation(Evaluation):
    def __init__(self, evaluate_on='valid', n_pictures=10, sample_images=False, sample_latents=False):
        # Consider the dataset labeled with the specified name (names are defined in the dataset configuration file).
        self.dataset = None
        self.evaluate_on = evaluate_on
        self.n_pictures = n_pictures
        self.sample_images = sample_images
        self.sample_latents = sample_latents


    def sample_new_images(self):
        # sample the required number of pictures randomly
        ids = np.random.choice(len(self.dataset), self.n_pictures)
        images_batch = torch.cat([self.dataset[id]['x'].unsqueeze(0) for id in ids])

        return images_batch

    def evaluate(self, optimization: pl.LightningModule) -> LogEntry:
        if self.dataset is None:
            self.dataset = optimization.data[self.evaluate_on]

        model = optimization.model

        # Check that the model has a definition of a method to reconstruct the inputs
        if not hasattr(model, 'reconstruct'):
            raise Exception('The model %s must implement a reconstruct(x) method with `x` as a picture' %
                            (model.__class__.__name__))

        # If the images are not sampled dynamically, pick the first n_pictures from the dataset
        if not self.sample_images:
            x = torch.cat([self.dataset[id]['x'].unsqueeze(0) for id in range(self.n_pictures)])
        # Otherwise pick random ones
        else:
            ids = np.random.choice(len(self.dataset), self.n_pictures)
            x = torch.cat([self.dataset[id]['x'].unsqueeze(0) for id in ids])

        # Move the images to the correct device
        device = next(model.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            # Compute the reconstructions
            x_rec = model.reconstruct(x).to('cpu')

            # Concatenate originals and reconstructions
            x_all = torch.cat([x.to('cpu'), x_rec], 2)

        # return a LogEntry
        return LogEntry(
            data_type=IMAGE_ENTRY,                          # Type of the logged object, to be interpreted by the logger
            value=make_grid(x_all, nrow=self.n_pictures)    # Value to log
        )
