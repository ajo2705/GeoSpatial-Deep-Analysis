import numpy as np
import torch
from enum import Enum
import logging

from train_util.classifier_train_helper import multiclass_classifier_comparison

TOLERANCE = 0.5
CONFIG_FILE = "config.yml"
logger = logging.getLogger('cnn_logger')


class Trainer(Enum):
    Linear = 0
    Classifier = 1


def validate(criterion, device, model, val_loader, trainer_metrics, trainer=Trainer.Classifier):
    # Validate the model
    if trainer == Trainer.Classifier:
        validator = multiclass_classifier_comparison
    else:
        validator = linear_regression_comparison

    model.eval()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        for val_data, val_targets in val_loader:
            val_data = val_data.to(device)
            val_targets = val_targets.to(device)

            val_scores = model(val_data)
            val_loss = criterion(val_scores, val_targets)
            val_loss += val_loss.item()

            val_correct, val_total = validator(val_correct, val_scores, val_targets, val_total, trainer_metrics)

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total

    return val_loss, val_accuracy


def train_validate(criterion, optimizer, num_epochs, device, model, train_loader, val_loader):
    for epoch in range(num_epochs):
        train_and_validate(criterion, device, epoch, model, num_epochs, optimizer, train_loader, val_loader)


def train_and_validate(criterion, device, epoch, model, num_epochs, optimizer, train_loader, val_loader,
                       trainer_metrics, trainer=Trainer.Classifier):
    loss_train = _single_epoch_train(criterion, device, model, optimizer, train_loader)
    loss = sum(loss_train)/len(loss_train)

    # Print loss for every epoch
    logger.info(f'Epoch [{epoch + 1}/{num_epochs}]; Train Loss: {loss:.4f}')

    # Validate the model
    val_loss, val_accuracy = validate(criterion, device, model, val_loader, trainer_metrics, trainer)
    logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    trainer_metrics.store_epoch_metrics(loss, val_loss, val_accuracy)

    return val_loss, val_accuracy


def train(criterion, device, model, num_epochs, optimizer, train_loader):
    for epoch in range(num_epochs):
        _single_epoch_train(criterion, device, model, optimizer, train_loader)


def _single_epoch_train(criterion, device, model, optimizer, train_loader):
    model.train()
    loss_train = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        scores = model(data)
        # loss = quantile_loss(scores, targets)
        loss = criterion(scores, targets)
        loss_train.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_train


def linear_regression_comparison(val_correct, val_scores, val_targets, val_total, train_metrics):
    tolerance = TOLERANCE
    val_predicted = np.squeeze(val_scores)
    val_total += val_targets.size(0)
    predicted = (torch.abs(val_predicted - val_targets) <= tolerance)
    val_correct += predicted.sum().item()

    train_metrics.compute_errors(val_predicted, val_targets, val_total)
    train_metrics.store_batch_pred_results(val_predicted, val_targets, predicted)
    return val_correct, val_total
