from code.evaluation.base import DatasetEvaluation


class ErrorComponentsEvaluation(DatasetEvaluation):
    def evaluate_batch(self, data, model):
        return model.compute_loss_components(data)


class ELBOEvaluation(ErrorComponentsEvaluation):
    def evaluate_batch(self, data, model):
        loss_components = model.compute_loss_components(data)
        return {'ELBO': -(loss_components['rec_loss'] + loss_components['reg_loss'])}
