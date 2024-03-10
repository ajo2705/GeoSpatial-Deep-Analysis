import os
import logging
import numpy as np
from sklearn.model_selection import train_test_split

from constants import Trainer, Base
from plot_util import plot_PCA, plot_distribution
from raster_util import BoundaryType, one_hot_encoding, store_data_files
from raster_converters import ClassifierRasterConverter

PATCH_SIZE = 10
NUM_CLASSES = 3

TRAIN_DATA = Trainer.TRAIN_DATA
TEST_DATA = Trainer.TEST_DATA
VALIDATION_DATA = Trainer.VALIDATION_DATA
IMAGE_FILE_PATH = Base.TRAINING_IMAGE_PATH

TRAIN_DATA_TARGET_FILEPATH = Trainer.TEMPLATE_DATA_PATH

logger = logging.getLogger('cnn_logger')


def load_process_data(raster_converter):
    non_outlier_target_values = raster_converter.remove_outliers(raster_converter.target_raster)

    raster_converter.plot_hist_data(bins=50)
    custom_boundary = [2.5, 7.5]
    raster_converter.generate_class_boundaries(BoundaryType.Percentile)
    logger.info("Trial Soil carbon class boundaries: ", raster_converter.class_boundaries)

    raster_converter.generate_class_boundaries(BoundaryType.Custom, custom_boundary)
    logger.info("Soil carbon class boundaries:",  raster_converter.class_boundaries)

    class_labels, image_patches = raster_converter.filter_datapoints()

    X = np.stack(image_patches)
    y = np.array(class_labels)

    logger.debug(f"MAX : {np.max(X)}  MIN : {np.min(X)}")

    # Split the dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    training_image = "train_data.jpeg"
    train_dist_image = "train_dist_data.jpeg"

    training_image_path = os.path.join(IMAGE_FILE_PATH, training_image)
    training_dist_image_path = os.path.join(IMAGE_FILE_PATH, train_dist_image)

    plot_PCA(X_train, y_train, training_image_path)
    plot_distribution(X_train, y_train, training_dist_image_path)

    logger.info("Training set class distribution:", np.bincount(y_train))
    logger.info("Validation set class distribution:", np.bincount(y_val))
    logger.info("Test set class distribution:", np.bincount(y_test))


    # Convert class labels to one-hot encoding
    y_test, y_train, y_val = one_hot_encoding(y_test, y_train, y_val, num_classes=NUM_CLASSES)

    return {TRAIN_DATA: (X_train, y_train),
            TEST_DATA: (X_test, y_test),
            VALIDATION_DATA: (X_val, y_val)}


def process_raster():
    covariate_rast_name = "CO_VARIATE_RASTER_NAME" # co_variate raster
    target_rast_name = "TARGET_RASTER" # target raster

    path = Base.BASE_RASTER_PATH
    raster_abs_path = os.path.join(path, covariate_rast_name)
    target_abs_path = os.path.join(path, target_rast_name)

    raster_converter = ClassifierRasterConverter(raster_abs_path, target_abs_path, PATCH_SIZE)
    data_set = load_process_data(raster_converter)

    path = os.path.join(Base.BASE_TRAINING_DATA_PATH, "classification_dataset")
    logger.info(f"Storage Path of pickled data: {path}")
    store_data_files(data_set, path)
