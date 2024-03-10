import torch
import torch.nn as nn


class Quantile(nn.Module):
    def __init__(self):
        super(Quantile, self).__init__()
        self.quantile = [0.25, 0.75]

    def set_quantile(self, quantiles):
        self.quantile = quantiles

    def forward(self, y_pred, y_true):
        losses = []
        for i, q in enumerate(self.quantile):
            errors = y_true - y_pred
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.log(torch.cosh(y_pred - y_true))
        return torch.mean(loss)


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, predictions, targets):

        weights = torch.exp(-torch.abs(targets - targets.mean()) / targets.std())

        # Calculate the weighted MSE
        loss = torch.mean(weights * (predictions - targets) ** 2)
        return loss


loss_map = {"MSE": nn.MSELoss,
            "MAE": nn.L1Loss,
            "Huber": nn.SmoothL1Loss,
            "Quantile": Quantile,
            "Log_Cosh": LogCoshLoss,
            "WeightMSE": WeightedMSELoss}


def get_loss_function(loss_str):
    return loss_map.get(loss_str, loss_map["MSE"])





