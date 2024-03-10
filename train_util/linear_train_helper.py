import os
import numpy as np
import torch
import logging
from libraries.constants import Base
from train_util.linear_train_metrics import LinearTrainMetrics
from train_util.model_trainer import train_and_validate, Trainer, TOLERANCE
from xlsx_write import write_linear_prediction_xlsx

logger = logging.getLogger('cnn_logger')


def train_validate_linear(criterion, optimizer, num_epochs, device, model, train_loader, val_loader, workbook,
                          tensor_log_name, early_stopper):
    linear_reg_metrics = LinearTrainMetrics(workbook, model, tensor_log_name=tensor_log_name)
    early_stopper.load_configs()

    linear_reg_metrics.store_data_distribution(train_loader, "Train")
    linear_reg_metrics.store_data_distribution(val_loader, "Validation")

    for epoch in range(num_epochs):
        linear_reg_metrics.prepare_next_epoch(epoch)
        val_loss, val_accuracy = train_and_validate(criterion, device, epoch, model, num_epochs, optimizer, train_loader, val_loader, linear_reg_metrics,
                                                    trainer=Trainer.Linear)

        early_stopper.update_counters(val_loss)
        if early_stopper.should_early_stop():
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            logger.info(f"Loading model configurations for {epoch - early_stopper.patience + 1}")
            break

    linear_reg_metrics.close_metrics()


def linear_regression_test_model(model, device, test_loader, loss_fn, workbook=None):
    """
        Model tester
        :param model:
        :param device:
        :param test_loader:
        :return:
        """
    model.eval()
    tolerance = TOLERANCE

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

            write_linear_prediction_xlsx(workbook, scores, targets, predicted, "Test")

        logger.info(f'Accuracy on test set: {(100 * correct / total):.2f}%')
        logger.info(f'MSE on test set: {(torch.tensor(ss_residual / total)) :.2f}')
        logger.info(f'RSquared on test set: {1 - (ss_residual / ss_total):.2f}')
    return


def get_model_save_file_loc(model_name, lr, bs, wd, loss_name):
    model_store_loc = os.path.join(Base.BASE_MODEL_STORE_PATH, f"{model_name}_lr={lr}_bs={bs}_wd={wd}_loss_name={loss_name}")
    if not os.path.exists(model_store_loc):
        os.makedirs(model_store_loc)

    model_name = "model.pt"
    model_file_name = os.path.join(model_store_loc, model_name)

    return model_file_name


def save_model(model, lr, bs, loss_name, wd):
    model_file_name = get_model_save_file_loc(type(model).__name__, lr, bs, wd, loss_name)

    # Save the torch model
    torch.save(model.state_dict(), model_file_name)
    logger.info(f"Saving model to : {model_file_name}")

