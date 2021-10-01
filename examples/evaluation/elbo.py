from framework.evaluation import DatasetEvaluation


class ErrorComponentsEvaluation(DatasetEvaluation):
    def evaluate_batch(self, data, model):
        return model.compute_loss_components(data)


class ELBOEvaluation(DatasetEvaluation):
    def evaluate_batch(self, data, model):

        loss_components = model.compute_loss_components(data)
        return {'ELBO': -(loss_components['reconstruction'] + loss_components['regularization'])}
