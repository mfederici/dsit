from framework.evaluation import DatasetEvaluation
import torch


class AccuracyEvaluation(DatasetEvaluation):
    def __init__(self, predict_params=None, **kwargs):
        super(AccuracyEvaluation, self).__init__(**kwargs)

        if predict_params is None:
            predict_params = dict()
        self.predict_params = predict_params

    def evaluate_batch(self, data, model):

        if not hasattr(model, 'predict'):
            raise Exception(
                'The trainer must implement a predict(x, **predict_params) method to use the Accuracy evaluation metric.'\
                'The predict function must return a discrete distribution object.')

        x = data['x']
        y = data['y'].squeeze().long()

        y_given_x = model.predict(x, **self.predict_params)

        y_pred = torch.argmax(y_given_x.probs, 1).squeeze().long()

        return {'Accuracy': (y == y_pred).float().mean().item()}


class CrossEntropyEvaluation(DatasetEvaluation):
    def __init__(self, predict_params=None, **kwargs):
        super(CrossEntropyEvaluation, self).__init__(**kwargs)

        if predict_params is None:
            predict_params = dict()
        self.predict_params = predict_params

    def evaluate_batch(self, data, model):

        if not hasattr(model, 'predict'):
            raise Exception(
                'The trainer must implement a predict(x, **predict_params) method to use the Accuracy evaluation metric.'\
                'The predict function must return a discrete distribution object.')

        x = data['x']
        y = data['y'].squeeze().long()

        y_given_x = model.predict(x, **self.predict_params)

        return {'CrossEntropy': -y_given_x.log_prob(y).mean().item() }