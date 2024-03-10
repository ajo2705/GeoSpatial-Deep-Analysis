import math

import torch
from torch.utils.tensorboard import SummaryWriter

from constants import Base
from xlsx_write import write_linear_prediction_xlsx, read_linear_prediction_xlsx


class LinearTrainMetrics:
    def __init__(self, workbook, model, tensor_log_name):
        self.loss_train = []
        self.epoch_loss = []
        self.loss_validation = []
        self.validation_accuracy = []

        self.workbook = workbook
        self.cur_sheet_name = None

        # ERROR METRICS
        self.mae_loss = 0
        self.mse_sum = 0
        self.mape_sum = 0
        self.ss_total = 0
        self.ss_residual = 0
        self.total_vals = 0
        self.tss_sum = 0
        self.rss_sum = 0

        self.torch_writer = SummaryWriter(Base.TENSOR_LOG_PATH + f"run_{tensor_log_name}/")
        self.epoch = 0
        self.model = model

    def compute_errors(self, outputs, targets, totals):
        self.mae_loss += torch.abs(outputs - targets).sum().item()
        self.mse_sum += torch.sum((outputs - targets) ** 2).item()

        perc_diff = (torch.abs(outputs - targets) / targets)
        self.mape_sum += torch.sum(perc_diff).item()

        # RSquared Error
        self.ss_total += torch.sum((targets - torch.mean(targets)) ** 2).item()
        self.ss_residual += torch.sum((targets - outputs) ** 2).item()

        # Explained Variance Score
        squared_diff_targets = (targets - torch.mean(targets)) ** 2
        squared_diff_outputs = (outputs - targets) ** 2
        self.tss_sum += torch.sum(squared_diff_targets).item()
        self.rss_sum += torch.sum(squared_diff_outputs).item()

        self.total_vals = totals

    def store_batch_pred_results(self, val_score, val_targets, predicted):
        write_linear_prediction_xlsx(self.workbook, val_score, val_targets, predicted, self.cur_sheet_name)

    def get_y_pred_data(self):
        return read_linear_prediction_xlsx(self.workbook, self.cur_sheet_name, col_name="Prediction")

    def get_y_target_data(self):
        return read_linear_prediction_xlsx(self.workbook, self.cur_sheet_name, col_name="TargetVal")

    def store_data_distribution(self, data_loader, data_type="train"):
        data_targets = []
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            data_targets.append(targets)

        data_targets = torch.cat(data_targets, dim=0)

        self.torch_writer.add_histogram(f'{data_type}/Target', data_targets, 0)

    def store_epoch_metrics(self, epoch_loss, val_loss, val_accuracy):
        self.torch_writer.add_scalar('Loss/train', epoch_loss, self.epoch )
        self.torch_writer.add_scalar('Accuracy/train', val_accuracy, self.epoch )
        self.torch_writer.add_scalar('Loss/validation', val_loss, self.epoch )

        self.torch_writer.add_histogram('Y_Pred', torch.tensor(self.get_y_pred_data()), self.epoch)
        self.torch_writer.add_histogram('Y_Target', torch.tensor(self.get_y_target_data()), self.epoch)

        # CNN MODEL METRICS
        for name, weight in self.model.named_parameters():
            self.torch_writer.add_histogram(name, weight, self.epoch)
            self.torch_writer.add_histogram(f'{name}.grad', weight.grad, self.epoch)

        # ERRORS
        mae = self.mae_loss / self.total_vals
        mse = self.mse_sum / self.total_vals
        mape = self.mape_sum / self.total_vals * 100
        r_squared = 1 - (self.ss_residual / self.ss_total)
        explained_variance_score = 1 - (self.rss_sum / self.tss_sum)

        self.torch_writer.add_scalar('Error_MAE', mae, self.epoch)
        self.torch_writer.add_scalar('Error_MSE', mse, self.epoch)
        self.torch_writer.add_scalar('Error_RMSE', math.sqrt(mse), self.epoch)
        self.torch_writer.add_scalar('Error_MAPE', mape, self.epoch)
        self.torch_writer.add_scalar('Error_RSQ', r_squared, self.epoch)
        self.torch_writer.add_scalar('Error_ExpVar', explained_variance_score, self.epoch)

    def prepare_next_epoch(self, epoch_num):
        self.cur_sheet_name = f'epoch_{epoch_num}'
        self.epoch = epoch_num
        self.torch_writer.flush()
        self.loss_train.clear()

        # ERROR METRICS
        self.refresh_error_metrics()

    def refresh_error_metrics(self):
        self.mae_loss = 0
        self.mse_sum = 0
        self.mape_sum = 0
        self.ss_total = 0
        self.ss_residual = 0
        self.total_vals = 0
        self.tss_sum = 0
        self.rss_sum = 0

    def close_metrics(self):
        self.torch_writer.flush()
        self.torch_writer.close()
