import os
import sys
from libraries.log_config import setup_logging
from raster_processor.raster_converter_classifier import process_raster as process_raster_classifier
from raster_processor.raster_converter_linear import process_raster as process_raster_linear
from raster_processor.raster_CNN import process_raster_CNN

# Put base path here
BASE_PATH = "BASE_PATH"

CNN_LIBRARY = "cnn"
LIB = "libraries"
RASTERS = "raster_processor"
MODEL_TRAINER = "model_trainers"

# BASE THINGS
lib_locations = [os.path.join(BASE_PATH, lib) for lib in [LIB, RASTERS, CNN_LIBRARY, MODEL_TRAINER]]
sys.path.extend(lib_locations)
setup_logging()

# For processing raster with classification
process_raster_CNN()





