import os
import pickle
import logging
import torch
from torch.utils.data import TensorDataset

from constants import Trainer, Base, ConfigParams

logger = logging.getLogger('cnn_logger')

def load_trains(base_path, load_set=Trainer.TRAIN_DATA):
    file_path = os.path.join(base_path, Trainer.TEMPLATE_DATA_PATH.format(filename=load_set))
    logger.info(f"Loading {load_set} from {file_path}")
    with open(file_path, "rb") as f:
        x, y = pickle.load(f)
        print(x.shape, y.shape)
        tensor_dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y))
    return tensor_dataset


def load_train_data():
    return load_trains(Base.BASE_TRAINING_DATA_PATH, Trainer.TRAIN_DATA)


def load_train_classifier_data():
    return load_trains(Base.CLASSIFIER_TRAINING_DATA_PATH, Trainer.TRAIN_DATA)


def load_train_linear_data():
    return load_trains(Base.LINEAR_TRAINING_DATA_PATH, Trainer.TRAIN_DATA)


def load_train_category_data(category):
    category_str = Trainer.TRAIN_DATA + "_cat_{}".format(category)
    return load_trains(Base.BASE_TRAINING_DATA_PATH, category_str)


def load_test_data():
    return load_trains(Base.BASE_TRAINING_DATA_PATH, Trainer.TEST_DATA)


def load_test_classifier_data():
    return load_trains(Base.CLASSIFIER_TRAINING_DATA_PATH, Trainer.TEST_DATA)


def load_test_linear_data():
    return load_trains(Base.LINEAR_TRAINING_DATA_PATH, Trainer.TEST_DATA)


def load_test_category_data(category):
    category_str = Trainer.TEST_DATA + "_cat_{}".format(category)
    return load_trains(Base.BASE_TRAINING_DATA_PATH, category_str)


def load_validation_data():
    return load_trains(Base.BASE_TRAINING_DATA_PATH, Trainer. VALIDATION_DATA)


def load_validation_classifier_data():
    return load_trains(Base.CLASSIFIER_TRAINING_DATA_PATH, Trainer. VALIDATION_DATA)


def load_validation_linear_data():
    return load_trains(Base.LINEAR_TRAINING_DATA_PATH, Trainer. VALIDATION_DATA)


def load_validation_category_data(category):
    category_str = Trainer.VALIDATION_DATA + "_cat_{}".format(category)
    return load_trains(Base.BASE_RESOURCE_PATH, category_str)