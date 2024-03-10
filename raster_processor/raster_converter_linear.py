import os
import logging
import numpy as np
from sklearn.model_selection import train_test_split

from constants import Trainer, Base
from raster_converters import LinearRasterConverter
from raster_util import store_data_files

PATCH_SIZE = 10

TRAIN_DATA = Trainer.TRAIN_DATA
TEST_DATA = Trainer.TEST_DATA
VALIDATION_DATA = Trainer.VALIDATION_DATA
IMAGE_FILE_PATH = Base.TRAINING_IMAGE_PATH

TRAIN_DATA_TARGET_FILEPATH = Trainer.TEMPLATE_DATA_PATH

logger = logging.getLogger('cnn_logger')


def load_process_data(raster_converter):
    raster_converter.filter_channels()

    raster_converter.normalise_classification_channels()
    target_data, image_patches = raster_converter.filter_datapoints()

    X = np.stack(image_patches)
    y = np.array(target_data)

    logger.debug(f"MAX : {np.max(X)}  MIN : {np.min(X)}")
    logger.debug(f"IMAGE PATCHES:: LEN -> {len(image_patches)}  SHAPE-> : {image_patches[0].shape}")

    # Split the dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    logger.info("Training set class distribution:", np.bincount(y_train))
    logger.info("Validation set class distribution:", np.bincount(y_val))
    logger.info("Test set class distribution:", np.bincount(y_test))

    return {TRAIN_DATA: (X_train, y_train),
            TEST_DATA: (X_test, y_test),
            VALIDATION_DATA: (X_val, y_val)}


def process_raster():
    covariate_rast_name = "CO_VARIATE_RASTER_NAME"
    target_rast_name = "TARGET_RASTER"

    path = Base.BASE_RASTER_PATH
    raster_abs_path = os.path.join(path, covariate_rast_name)
    target_abs_path = os.path.join(path, target_rast_name)

    raster_converter = LinearRasterConverter(raster_abs_path, target_abs_path, PATCH_SIZE)
    data_set = load_process_data(raster_converter)

    path = os.path.join(Base.BASE_TRAINING_DATA_PATH, "classification_dataset")
    logger.info(f"Storage Path of pickled data: {path}")
    store_data_files(data_set, path)
