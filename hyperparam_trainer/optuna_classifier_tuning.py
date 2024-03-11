import torch
import logging
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna

from train_util.model_trainer import train, validate
from cnn.param_classifier_cnn_model import ParameterizedCNN
from load_data import load_train_data, load_validation_data

batch_size = 128
num_epochs = 50

logger = logging.getLogger('cnn_logger')


def train_eval_dataset(hidden_size, learning_rate, kernel_size, device):
    model = ParameterizedCNN(3, hidden_size, kernel_size).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    train_loader = DataLoader(dataset=load_train_data(), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=load_validation_data(), batch_size=batch_size, shuffle=False)

    train(criterion, device, model, num_epochs, optimizer, train_loader)
    _, val_accuracy = validate(criterion, device, model, val_loader)

    return val_accuracy


def objective(trial):
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    kernel_size = trial.suggest_categorical('kernel_size', [2, 3, 4])

    # Evaluate the model with the current hyperparameters
    accuracy = train_eval_dataset(hidden_size, learning_rate, kernel_size, device)
    return accuracy


def optuna_trainer():
    torch.manual_seed(42)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    logger.info(f"Best params for optuna training {study.best_params}")
    logger.info(f"Accuracy of best model {study.best_value}")


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optuna_trainer()
