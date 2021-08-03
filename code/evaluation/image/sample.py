from torchvision.utils import make_grid
from code.evaluation import Evaluation
from code.loggers import LogEntry
from code.loggers.log_entry import IMAGE_ENTRY
from code.models.base import GenerativeModel
import pytorch_lightning as pl


class ImageSampleEvaluation(Evaluation):
    def __init__(self, n_pictures=10, sampling_params=None):
        self.n_pictures = n_pictures
        self.sampling_params = sampling_params if not(sampling_params is None) else dict()

    def evaluate(self, optimization: pl.LightningModule) -> LogEntry:
        model = optimization.model

        assert isinstance(model, GenerativeModel)

        x_gen = model.sample([self.n_pictures], **self.sampling_params).to('cpu')

        return LogEntry(
            data_type=IMAGE_ENTRY,                          #Type of the logged object, to be interpreted by the logger
            value=make_grid(x_gen, nrow=self.n_pictures)    # Value to log
        )
