import torch

from xlsx_write import write_classifer_prediction_xlsx


def multiclass_classifier_test_model(model, device, test_loader, workbook=None):
    """
    Model tester
    :param model:
    :param device:
    :param test_loader:
    :return:
    """
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        correct = 0
        total = 0
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            scores = model(data)
            predicted = torch.argmax(scores, 1)
            predictions.extend(predicted.data.cpu().numpy())
            val_targets = torch.argmax(targets, dim=1)
            true_labels.extend(val_targets.data.cpu().numpy())
            total += val_targets.size(0)

            correct += (predicted == val_targets).sum().item()

            write_classifer_prediction_xlsx(workbook, scores, val_targets, predicted)

        print(f'Accuracy on test set: {(100 * correct / total):.2f}%')
    return predictions, true_labels


def multiclass_classifier_comparison(val_correct, val_scores, val_targets, val_total):
    val_predicted = torch.argmax(val_scores, dim=1)
    target_vals = torch.argmax(val_targets, dim=1)
    val_total += target_vals.size(0)
    val_correct += (val_predicted == target_vals).sum().item()
    return val_correct, val_total
