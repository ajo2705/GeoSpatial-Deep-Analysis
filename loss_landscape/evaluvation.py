"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import torch


def eval_loss(device, model, criterion, loader):
    model.eval()

    with torch.no_grad():
        val_correct = 0
        val_total = 0
        loss = 0
        tolerance = 1
        for val_data, val_targets in loader:
            val_data = val_data.to(device)
            val_targets = val_targets.to(device)

            val_scores = model(val_data)
            val_loss = criterion(val_scores, val_targets)
            loss += val_loss.item()

            val_total += val_targets.size(0)
            predicted = (abs(val_scores - val_targets) <= tolerance)

            val_correct += predicted.sum().item()

        val_loss /= val_total
        val_accuracy = 100 * val_correct / val_total

    return val_loss, val_accuracy
