import torch
import numpy as np
from torchvision.utils import make_grid
from code.evaluation import Evaluation
from code.loggers import LogEntry
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.distributions import Independent

from code.loggers.log_entry import PLOT_ENTRY


class ImageLandmarksEvaluation(Evaluation):
    def __init__(self, evaluate_on='valid', n_pictures=10, sample_images=False, sample_latents=False, padding=None, landmark_color='b'):
        # Consider the dataset labeled with the specified name (names are defined in the dataset configuration file).
        self.dataset = None
        self.evaluate_on = evaluate_on
        self.n_pictures = n_pictures
        self.sample_latents = sample_latents
        self.sample_images = sample_images
        self.padding = [0, 0] if padding is None else padding
        self.landmark_color = landmark_color


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
        if not hasattr(model, 'predict'):
            raise Exception('The model %s must implement a predict(x)->y method with `x` as a picture and `y` as a list of landmarks' %
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

        # Compute the landmark locations
        p_y_given_x = model.predict(x)

        #if isinstance(p_y_given_x, Independent):
        #    p_y_given_x = p_y_given_x.base_dist

        f, ax = plt.subplots(1, self.n_pictures, figsize=(5*self.n_pictures,5))

        for i in range(self.n_pictures):
            ax[i].imshow(x[i].permute(1, 2, 0).data.to('cpu') )
            mu = p_y_given_x.mean[i].to('cpu').reshape(-1, 2).data.to('cpu')
            sigma = p_y_given_x.stddev[i].to('cpu').reshape(-1, 2).data.to('cpu')
            ax[i].plot(mu[:, 0] + self.padding[0],
                       mu[:, 1] + self.padding[1],
                       'o', color=self.landmark_color)
            ax[i].axis('off')
            for j in range(mu.shape[0]):
                ax[i].fill_between([
                    mu[j, 0] + self.padding[0] - sigma[j,0],
                    mu[j, 0] + self.padding[0] + sigma[j,0]
                ],
                    mu[j, 1] + self.padding[1] - sigma[j, 1]
                ,
                    mu[j, 1] + self.padding[1] + sigma[j, 1]
                ,
                    color=self.landmark_color, alpha=0.5
                )

        # return a LogEntry
        return LogEntry(
            data_type=PLOT_ENTRY,                       # Type of the logged object, to be interpreted by the logger
            value=f                                     # Value to log
        )
