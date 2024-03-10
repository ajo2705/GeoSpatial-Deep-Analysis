"""
All constants used across all files
"""

class Trainer:
    TRAIN_DATA = "train"
    TEST_DATA = "test"
    VALIDATION_DATA = "valid"

    TEMPLATE_DATA_PATH = "{filename}_data.pickle"


class Base:
    BASE_PATH = "{base_path}"   # Directory location of resources and all other base folders. Default set it to "/". Configure in config file
    BASE_RESOURCE_PATH = BASE_PATH + "Resources/"
    BASE_IMAGE_PATH = BASE_RESOURCE_PATH + "Images/"
    BASE_XLSX_PATH = BASE_RESOURCE_PATH + "Tables/"
    BASE_RASTER_PATH = BASE_RESOURCE_PATH + "Rasters/"
    BASE_MODEL_PATH = BASE_RESOURCE_PATH + "Models/"

    BASE_TRAINING_DATA_PATH = BASE_RESOURCE_PATH + "Training/"
    CLASSIFIER_TRAINING_DATA_PATH = BASE_TRAINING_DATA_PATH + "classification_dataset/"
    LINEAR_TRAINING_DATA_PATH = BASE_TRAINING_DATA_PATH + "linear_dataset/"

    RASTER_IMAGE_PATH = BASE_RASTER_PATH + "images/"
    TRAINING_IMAGE_PATH = BASE_IMAGE_PATH + "training_data/"

    BASE_LOG_PATH = BASE_PATH + "Logs/"
    TENSOR_LOG_PATH = BASE_LOG_PATH + "tensorLog/"
    HYPERPARAM_LOG_PATH = BASE_LOG_PATH + "hyperparam/run3/tensorLog/"

    BASE_MODEL_STORE_PATH = BASE_MODEL_PATH + "trained_models/"
    BASE_HYPER_MODEL_STORE_PATH = BASE_MODEL_PATH + "hyper_run/"


class ConfigParams:
    CONFIG_FILE = "config.yml"

    # Model parameters
    CNN = "CNN"

    loss = CNN + "/loss"
    hyperparameters = CNN + "/hyperparameters"

    num_epochs = hyperparameters + "/num_epochs"
    batch_size = hyperparameters + "/batch_size"
    learning_rate = hyperparameters + "/learning_rate"

    early_stopper = CNN + "/early_stopper"
    patience = early_stopper + "/patience"

    model_details = CNN + "/model"
    model_file = model_details + "/file"
    model_class = model_details + "/class"
    model_config = model_details + "/config"

    # Loss landscape parameters
    loss_landscape = "LossLandscape"

    plot = loss_landscape + "/plot"
    direction = plot + "/direction"

    x_coord = direction + "/x"
    y_coord = direction + "/y"
    dir_file = direction + "/dir_file"
    dir_type = direction + "/dir_type"
    x_norm = direction + "/xnorm"
    y_norm = direction + "/ynorm"
    x_ignore = direction + "/xignore"
    y_ignore = direction + "/yignore"
    same_dir = direction + "/same_dir"
    surf_file = direction + "/surf_file"

    trained_model = loss_landscape + "/trained_model"
    file1 = trained_model + "/file1"
    file2 = trained_model + "/file2"
    file3 = trained_model + "/file3"


