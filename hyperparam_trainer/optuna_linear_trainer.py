import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import optuna
import yaml
import openpyxl as xl

from cnn.param_linear_reg_cnn_model import ParameterizedCNN
from load_data import load_train_linear_data, load_validation_linear_data, load_test_linear_data
from libraries.constants import Base, ConfigParams
from loss_functions import get_loss_function
from train_util.early_stopper import EarlyStopper
from train_util.hyper_param_helper import train_validate_hyperparam_trainer, test_hyperparam
from train_util.hyperparam_metrices import HyperParameters, HyperLinearMetrics
from xlsx_write import get_xlsx_write_file

CONFIG_FILE = "config.yml"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger('cnn_logger')


def get_config_parameter(config, parameter_name):
    content = config
    for val in parameter_name.split("/"):
        content = content[val]

    return content


def load_configurations():
    res_path = Base.BASE_RESOURCE_PATH
    with open(os.path.join(res_path + CONFIG_FILE)) as f:
        configurations = yaml.safe_load(f)

    return configurations


def get_loss(config_dict):
    loss = get_config_parameter(config_dict, ConfigParams.loss)
    return loss, get_loss_function(loss)


def get_constant_hyperparameters(config_dict):
    batch_size = get_config_parameter(config_dict, ConfigParams.batch_size)
    num_epochs = get_config_parameter(config_dict, ConfigParams.num_epochs)

    return batch_size, num_epochs


def get_input_channels(config_dict):
    model_config = get_config_parameter(config_dict, ConfigParams.model_config)
    input_channels = model_config.get("input_channels", 97)

    return input_channels


def train_linaar_eval_dataset(hidden_size, learning_rate, kernel_size, device):
    hyperparameters = set_hyperparameters(hidden_size, learning_rate, kernel_size)

    config_dict = load_configurations()
    batch_size, num_epochs = get_constant_hyperparameters(config_dict)
    input_channels = get_input_channels(config_dict)
    model = ParameterizedCNN(input_channels, hidden_size, kernel_size).to(device)

    # Define loss function and optimizer
    loss_name, loss_function = get_loss(config_dict)  # nn.MSELoss()
    criterion = loss_function()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_loader = DataLoader(dataset=load_train_linear_data(), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=load_validation_linear_data(), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=load_test_linear_data(), batch_size=batch_size, shuffle=False)

    # Create an Excel workbook and add a worksheet
    xlsx_file = get_xlsx_write_file(trainer="optuna_3")
    logger.info(f"Writing xlsx into file {xlsx_file}")
    workbook = xl.Workbook()

    tensor_log_name = f'hidden_size={hidden_size}_lr={learning_rate}_kernel_size={kernel_size}, loss_name={loss_name}'
    hyper_reg_metrics = HyperLinearMetrics(workbook, model, tensor_log_name=tensor_log_name)
    hyper_reg_metrics.set_hyperparameters(hyperparameters)

    early_stopper = EarlyStopper(model, f"Linear_Optuna_{tensor_log_name}")

    train_validate_hyperparam_trainer(criterion, optimizer, num_epochs, device, model, train_loader, val_loader,
                                      early_stopper, hyper_reg_metrics)
    early_stopper.load_saved_model()
    acc, rsq = test_hyperparam(model, device, test_loader, workbook)
    hyper_reg_metrics.prepare_next_config(acc, rsq)
    return rsq


def objective_linear(trial):
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    kernel_size = trial.suggest_categorical('kernel_size', [2, 3, 4])

    # Evaluate the model with the current hyperparameters
    accuracy = train_linaar_eval_dataset(hidden_size, learning_rate, kernel_size, device)
    return accuracy


def set_hyperparameters(hidden_size, learning_rate, kernel_size):
    hyperparameters = HyperParameters()
    hyperparameters.kernel_size = kernel_size
    hyperparameters.learning_rate = learning_rate
    hyperparameters.hidden_size = hidden_size
    return hyperparameters


def optuna_trainer():
    torch.manual_seed(42)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective_linear, n_trials=1000)

    logger.info(f"Best params for optuna training {study.best_params}")
    logger.info(f"Accuracy of best model {study.best_value}")


# Set device
def main():
    optuna.logging.enable_propagation()
    optuna_trainer()