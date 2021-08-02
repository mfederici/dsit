from torchvision.utils import make_grid
from code.evaluation import Evaluation
from code.models.base import GenerativeModel


class ImageSampleEvaluation(Evaluation):
    def __init__(self, n_pictures=10, **kwargs):
        self.n_pictures = n_pictures
        self.kwargs = kwargs

    def evaluate(self, optimization):
        model = optimization.model

        assert isinstance(model, GenerativeModel)

        x_gen = model.sample([self.n_pictures], **self.kwargs).to('cpu')

        # Concatenate originals and reconstructions
        return {
            'type': 'figure',  # Type of the logged object, to be interpreted by the logger
            'value': make_grid(x_gen, nrow=self.n_pictures),  # Value to log
        }