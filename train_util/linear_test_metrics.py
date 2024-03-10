class LinearTestMetrices:
    def __init__(self, model, loss_fn):
        self.loss_test = []
        self.model = model
        self.loss_fn = loss_fn

    def compute_on_batch_prediction(self, predicted, targets):
        self.loss_test.extend(self.loss_fn(predicted, targets))