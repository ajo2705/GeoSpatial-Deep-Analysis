import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from math import ceil
from enum import Enum
from matplotlib import pyplot as plt

from libraries.constants import Base, Trainer
from libraries.plot_util import save_fig


# Create a function to detect outliers based on IQR
def detect_outliers_iqr(data, q1=0.25, q3=0.75):
    q1_value = np.percentile(data, q1 * 100)
    q3_value = np.percentile(data, q3 * 100)
    iqr = q3_value - q1_value
    outliers = np.where((data < (q1_value - 1.5 * iqr)) | (data > (q3_value + 1.5 * iqr)))
    return outliers


def percentile_based_boundary(target_values):
    # Change the class boundaries percentile according to need.
    class_boundaries = np.percentile(target_values, [33, 66])

    # 1 is added to max for safety; proper working of digitize
    class_boundaries = np.concatenate(([0, ceil(np.max(target_values) + 1)], class_boundaries))
    class_boundaries = np.sort(class_boundaries)

    return class_boundaries


def one_hot_encoding(y_test, y_train, y_val, num_classes):
    y_train_torch = torch.from_numpy(y_train)
    y_test_torch = torch.from_numpy(y_test)
    y_val_torch = torch.from_numpy(y_val)

    y_train = F.one_hot(y_train_torch, num_classes).float()
    y_test = F.one_hot(y_test_torch, num_classes).float()
    y_val = F.one_hot(y_val_torch, num_classes).float()
    return y_test, y_train, y_val


def plot_data_non_outliers(non_outlier_targets, bins):
    hist_plot = f"hist_plot_{bins}.jpeg"
    plt.hist(non_outlier_targets, bins=bins)

    hist_plot_image_path = os.path.join(Base.TRAINING_IMAGE_PATH, hist_plot)
    save_fig(hist_plot_image_path)


def store_data_files(data_dict, base_path, logger=None):
    for key, val in data_dict.items():
        abs_path = _get_file_handle(base_path, key)

        if logger:
            logger.info(f"Writing into file {abs_path}")

        with open(abs_path, "wb") as handle:
            pickle.dump(val, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _get_file_handle(base_path, key):
    file_path = Trainer.TEMPLATE_DATA_PATH.format(filename=key)
    abs_path = os.path.join(base_path, file_path)
    if not (os.path.exists(abs_path)):
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    return abs_path


class BoundaryType(Enum):
    Percentile = 0
    Custom = 1


