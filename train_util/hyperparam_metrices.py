from torch.utils.tensorboard import SummaryWriter
from train_util.linear_train_metrics import LinearTrainMetrics
from libraries.constants import Base


class HyperParameters:
    def __init__(self):
        self._hidden_size = 0
        self._lr = 0
        self._kernel_size = 0

    @property
    def hidden_size(self):
        return self._hidden_size

    @hidden_size.setter
    def hidden_size(self, hs):
        self._hidden_size = hs

    @property
    def learning_rate(self):
        return self._lr

    @learning_rate.setter
    def learning_rate(self, lr):
        self._lr = lr

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, kernel_size):
        self._kernel_size = kernel_size


class HyperLinearMetrics(LinearTrainMetrics):
    def __init__(self, workbook, model, tensor_log_name):
        super(HyperLinearMetrics, self).__init__(workbook, model, tensor_log_name)
        self.hyperparameters = HyperParameters()
        self.torch_writer = SummaryWriter(Base.HYPERPARAM_LOG_PATH + f"run_{tensor_log_name}/")

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters.hidden_size = hyperparameters.hidden_size
        self.hyperparameters.learning_rate = hyperparameters.learning_rate
        self.hyperparameters.kernel_size = hyperparameters.kernel_size

    def prepare_next_config(self, acc, rsq):
        self.torch_writer.add_hparams(
            {
                "hidden_size": self.hyperparameters.hidden_size,
                "learning_rate": self.hyperparameters.learning_rate,
                "kernel_size": self.hyperparameters.kernel_size
            },
            {
                "accuracy": acc,
                "r_squared": rsq
            }
        )

        self.torch_writer.flush()

    def close_metrics(self):
        super(HyperLinearMetrics, self).close_metrics()