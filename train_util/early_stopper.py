import os

import torch
import yaml

from constants import Base, ConfigParams
from train_util.model_trainer import CONFIG_FILE


class EarlyStopper(object):
    MODEL_SAVE_PATH_TEMPLATE = Base.BASE_HYPER_MODEL_STORE_PATH + "{}_best_model.pt"

    def __init__(self, model, save_model_qualifier):
        self.patience = 20
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        self.model = model
        self.model_save_path = self.MODEL_SAVE_PATH_TEMPLATE.format(save_model_qualifier)

    def load_configs(self):
        configs = self._get_config()
        self.patience = self.get_config_parameter(configs, ConfigParams.patience)

    def update_counters(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0

            # Save the model checkpoint
            torch.save(self.model.state_dict(), self.model_save_path)
        else:
            self.early_stopping_counter += 1

    def should_early_stop(self):
        return self.early_stopping_counter >= self.patience

    @staticmethod
    def get_config_parameter(config, parameter_name):
        content = config
        for val in parameter_name.split("/"):
            content = content[val]

        return content

    @staticmethod
    def _get_config():
        res_path = Base.BASE_RESOURCE_PATH
        with open(os.path.join(res_path + CONFIG_FILE)) as f:
            configurations = yaml.safe_load(f)

        return configurations

    def load_saved_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path))
