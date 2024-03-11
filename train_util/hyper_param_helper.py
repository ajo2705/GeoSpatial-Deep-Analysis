import torch
import numpy as np
import logging
from train_util.hyperparam_metrices import HyperLinearMetrics
from train_util.model_trainer import train_and_validate, Trainer
from libraries.xlsx_write import write_linear_prediction_xlsx

logger = logging.getLogger('cnn_logger')


def train_validate_hyperparam_trainer(criterion, optimizer, num_epochs, device, model, train_loader, val_loader,
                          early_stopper, hyper_metrics):

    early_stopper.load_configs()

    for epoch in range(num_epochs):
        hyper_metrics.prepare_next_epoch(epoch)
        val_loss, val_accuracy = train_and_validate(criterion, device, epoch, model, num_epochs, optimizer,
                                                    train_loader, val_loader, hyper_metrics,
                                                    trainer=Trainer.Linear)

        early_stopper.update_counters(val_loss)
        if early_stopper.should_early_stop():
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            logger.info(f"Loading model configurations for {epoch - early_stopper.patience + 1}")
            break

    # self.add_hparams
    # linear_reg_metrics.close_metrics()


def test_hyperparam(model, device, test_loader, workbook=None):
    """
            Model tester
            :param model:
            :param device:
            :param test_loader:
            :return:
            """
    model.eval()
    tolerance = 0.3

    with torch.no_grad():
        correct = 0
        total = 0
        ss_total = 0
        ss_residual = 0
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            val_scores = model(data)
            scores = np.squeeze(val_scores)
            total += targets.size(0)
            predicted = (abs(scores - targets) <= tolerance)

            correct += predicted.sum().item()
            ss_total += torch.sum((targets - torch.mean(targets)) ** 2).item()
            ss_residual += torch.sum((targets - scores) ** 2).item()

            if workbook:
                write_linear_prediction_xlsx(workbook, scores, targets, predicted, "Test")
            # print(list(zip(predicted_list, val_list)))
        logger.info(f'Accuracy on test set: {(100 * correct / total):.2f}%')
        logger.info(f'RSquared on test set: {1 - (ss_residual / ss_total):.2f}')

    return (100 * correct / total), 1 - (ss_residual / ss_total)