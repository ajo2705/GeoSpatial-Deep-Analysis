import os
import sys
from libraries.log_config import setup_logging
# RASTER PROCESSING IMPORTS
from raster_processor.raster_converter_classifier import process_raster as process_raster_classifier
from raster_processor.raster_converter_linear import process_raster as process_raster_linear
from raster_processor.raster_CNN import process_raster_CNN

# SINGLE MODEL RUN IMPORTS
from model_trainers.train_linear import train_model

# HYPERPARAMETER TRAINING IMPORTS
from hyperparam_trainer.optuna_linear_trainer import main as optuna_training

# Put base path here
BASE_PATH = "BASE_PATH"

CNN_LIBRARY = "cnn"
LIB = "libraries"
RASTERS = "raster_processor"
MODEL_TRAINER = "model_trainers"
HYPERPARAM_LIBRARY = "hyperparameter_trainer"

# BASE THINGS
lib_locations = [os.path.join(BASE_PATH, lib) for lib in [LIB, RASTERS, CNN_LIBRARY, HYPERPARAM_LIBRARY, MODEL_TRAINER]]
sys.path.extend(lib_locations)
setup_logging()

# For processing raster with classification
process_raster_CNN()





