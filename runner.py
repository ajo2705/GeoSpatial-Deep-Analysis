import os
import sys
from libraries.log_config import setup_logging
from raster_processor.raster_converter_classifier import process_raster as process_raster_classifier
from raster_processor.raster_converter_linear import process_raster as process_raster_linear

# Put base path here
BASE_PATH = "BASE_PATH"

LIB = "libraries"
RASTERS = "raster_processor"

# BASE THINGS
lib_locations = [os.path.join(BASE_PATH, lib) for lib in [LIB, RASTERS]]
sys.path.extend(lib_locations)
setup_logging()

# For processing raster with classification
process_raster_classifier()

#For processing raster for linear regression
process_raster_linear()





