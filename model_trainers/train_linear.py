import os
import torch
import yaml
import importlib
import logging

import torch.optim as optim
from torch.utils.data import DataLoader

from libraries.constants import Base, ConfigParams
from train_util.linear_train_helper import train_validate_linear, linear_regression_test_model, save_model
from train_util.early_stopper import EarlyStopper
from libraries.xlsx_write import get_xlsx_write_file
from load_data import load_train_linear_data, load_validation_linear_data, load_test_linear_data

import openpyxl as xl

from loss_functions import get_loss_function

CONFIG_FILE = "config.yml"
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


def get_hyperparameters(config_dict):
    batch_size = get_config_parameter(config_dict, ConfigParams.batch_size)
    num_epochs = get_config_parameter(config_dict, ConfigParams.num_epochs)
    learning_rate = get_config_parameter(config_dict, ConfigParams.learning_rate)

    return batch_size, num_epochs, learning_rate


def get_loss(config_dict):
    loss = get_config_parameter(config_dict, ConfigParams.loss)
    return loss, get_loss_function(loss)

def get_nn_config(config_dict):
    config = get_config_parameter(config_dict, ConfigParams.model_config)
    return config

def load_model(config_dict, device):
    module_name = get_config_parameter(config_dict, ConfigParams.model_file)
    class_name = get_config_parameter(config_dict, ConfigParams.model_class)

    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    model_config = get_nn_config(config_dict)
    model = model_class(model_config).to(device)
    return model


def train_model():
    config_dict = load_configurations()
    batch_size, num_epochs, learning_rate = get_hyperparameters(config_dict)

    torch.manual_seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(config_dict, device)
    logger.info("****** MODEL DETAILS *******")
    logger.info(model)
    logger.info("*********  MODEL  **********")

    # Define loss function and optimizer
    loss_weights = torch.tensor([3.5, 1.2, 1.9])

    loss_name, loss_function = get_loss(config_dict)# nn.MSELoss()
    criterion = loss_function()

    weight_decay = 1e-2
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader = DataLoader(dataset=load_train_linear_data(), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=load_validation_linear_data(), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=load_test_linear_data(), batch_size=batch_size, shuffle=False)

    # Create an Excel workbook and add a worksheet
    xlsx_file = get_xlsx_write_file(trainer="trainer")
    logging.info(f"Writing xlsx into file {xlsx_file}")
    workbook = xl.Workbook()

    run_number = xlsx_file.split("_")[-1].split(".")[0]
    tensor_log_name = f"{run_number}_loss={loss_name}_filtered=true"

    # Train the model
    early_stopper = EarlyStopper(model, "Linear")
    train_validate_linear(criterion, optimizer, num_epochs, device, model, train_loader, val_loader, workbook,
                          tensor_log_name, early_stopper)

    # Test the model
    early_stopper.load_saved_model()
    linear_regression_test_model(model, device, test_loader, criterion, workbook)
    workbook.save(xlsx_file)

    #Save model
    save_model(model, learning_rate, batch_size, loss_name, weight_decay)
