import os
import logging
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split

from libraries.constants import Trainer, Base
from libraries.raster_util import store_data_files


PATCH_SIZE = 10
CELL_SIZE = 0.01
GRID_SIZE = 10
TRAIN_DATA = Trainer.TRAIN_DATA
TEST_DATA = Trainer.TEST_DATA
VALIDATION_DATA = Trainer.VALIDATION_DATA
IMAGE_FILE_PATH = Base.TRAINING_IMAGE_PATH

TRAIN_DATA_TARGET_FILEPATH = Trainer.TEMPLATE_DATA_PATH
target = "TARGET"
FILTER_CHANNELS = ["soil_taxonomy_refined", "LC_type1", "CDL_1km", 'elev', 'srad', 'slope', 'local_upslope_curvature',
                   'local_downslope_curvature', 'aspect', 'bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_15',
                   'NDVI_p50', 'EVI_p0', 'Final_sm', 'clay_b0', 'clay_b10', 'ai', 'et0', 'SBIO_7', '5to15cm_SBIO_3',
                   '5to15cm_SBIO_4', '5to15cm_SBIO_5', 'crop_land']

logger = logging.getLogger('cnn_logger')


def one_hot_encode(co_variate_raster, band_names=None):
    """
    one hot encoding on bands with discrete values and return new raster.
    :param co_variate_raster:
    :return:
    """
    all_one_hot_encoded_bands = []
    normalized_bands = []

    # DEBUG
    encoded_band_name = []
    normal_band_name = []
    # DEBUG

    for idx, band in enumerate(co_variate_raster):
        # Since band are normalized, bands with values greater than 1 should be classfication bands
        if np.nanmax(band) > 1.0:
            band[band == -9.999e+03] = np.nan
            discrete_values = np.unique(band[~np.isnan(band)])
            num_discrete_values = len(discrete_values)
            value_to_index = {value: index for index, value in enumerate(discrete_values)}

            one_hot_encoded_band = np.zeros((num_discrete_values,) + band.shape)
            for value in discrete_values:
                index = value_to_index[value]
                one_hot_encoded_band[index, :, :] = (band == value)

                # DEBUG
                if band_names:
                    encoded_band_name.append(f"{band_names[idx]}_{value}")
                # END DEBUG

            all_one_hot_encoded_bands.append(one_hot_encoded_band.astype('float32'))

        else:
            normalized_bands.append(band)
            # DEBUG
            if band_names:
                normal_band_name.append(band_names[idx])
            # END DEBUG

    one_hot_encoded_bands = np.concatenate(all_one_hot_encoded_bands, axis=0)
    normalized_bands = np.stack(normalized_bands, axis=0)

    logger.info(f"Shape of one hot encoded bands : {one_hot_encoded_bands.shape}")
    logger.info(f"Shape of normalized bands : {normalized_bands.shape}")

    result = np.concatenate((one_hot_encoded_bands, normalized_bands), axis=0)
    logger.info(f"Result shape after one hot encoding {result.shape}")
    encoded_band_name.extend(normal_band_name)

    # DEBUG
    if band_names:
        return result, encoded_band_name
    # END DEBUG

    return result


def handle_outliers(target_raster):
    non_nan_target_values = target_raster[~np.isnan(target_raster)]

    q1_value = np.percentile(non_nan_target_values, 25)
    q3_value = np.percentile(non_nan_target_values, 75)

    iqr = q3_value - q1_value
    lower_bound = q1_value - 1.5 * iqr
    upper_bound = q3_value + 1.5 * iqr

    logger.debug(f"q1 value : {q1_value} \t q3 value : {q3_value}")
    logger.debug(f"Lower bound : {lower_bound} \t Upper bound : {upper_bound}")

    target_raster[
        (target_raster < lower_bound) | (target_raster > upper_bound)] = np.nan  # np.median(non_nan_target_values
    return target_raster


def get_raster_data(raster_abs_path, target_abs_path):
    """
    Read covariate and target raster
    :param raster_abs_path:
    :param target_abs_path:
    :return: Tuple of covariate and target raster
    """

    with rasterio.open(raster_abs_path) as covar_raster_file:
        co_variate_raster = covar_raster_file.read()
        no_data_band_index = [idx for idx in range(1, covar_raster_file.count + 1)
                              if covar_raster_file.tags(idx)['NAME'] == 'NoData_Mask'][0] - 1
        no_data_mask = co_variate_raster[no_data_band_index].astype(bool)

        # Uncomment this for filtering bands
        co_variate_raster = filter_bands(covar_raster_file)

    with rasterio.open(target_abs_path) as tgt_raster_file:
        target_raster = tgt_raster_file.read(1)

        # Set the regions with NoData in co_variate as nan in target raster
        target_raster[no_data_mask] = np.nan

        # Remove no_data_mask from co_variate_raster
        # Comment this for filtering bands
        # co_variate_raster = np.concatenate(
        #     (co_variate_raster[:no_data_band_index], co_variate_raster[no_data_band_index + 1:]), axis=0)

    return co_variate_raster, target_raster


def extract_data_from_co_variate_target(co_variate_raster, target_raster):
    image_patches = []
    targets = []

    height, width = target_raster.shape
    patch_size = 20
    half_patch_size = patch_size // 2

    counter = 0  # DEBUG
    # Find points in target raster that have valid value and create an X patch
    for y in range(half_patch_size, height - half_patch_size):
        for x in range(half_patch_size, width - half_patch_size):
            tgt_val = target_raster[y, x]

            if np.isnan(tgt_val):
                continue

            patch = co_variate_raster[:, y - half_patch_size: y + half_patch_size,
                    x - half_patch_size: x + half_patch_size]
            if np.isnan(patch).any() or -9999.0 in patch:
                counter += 1
                continue

            image_patches.append(patch)
            targets.append(tgt_val)

    x = np.stack(image_patches)
    y = np.array(targets)

    logger.debug(x.shape)
    logger.debug(y.shape)
    logger.debug(f"SKIPPED {counter}")  # DEBUG

    return x, y


def filter_bands(co_variate_raster):
    class_datas = [co_variate_raster.tags(idx)['NAME'] for idx in range(1, co_variate_raster.count + 1)]
    index_list = [index for index, item in enumerate(class_datas) if item in FILTER_CHANNELS]

    raster_data = co_variate_raster.read()
    total_channels = [raster_data[i, :, :] for i in index_list]
    co_variate_raster = np.stack(total_channels, axis=0)

    return co_variate_raster


def load_process_data(raster_abs_path, target_abs_path):
    co_variate_raster, target_raster = get_raster_data(raster_abs_path, target_abs_path)
    # We use normalized co_variate_raster, no need to further normalize bands
    # normalize_co_variate_raster(co_variate_raster)

    encoded_co_variate_raster = one_hot_encode(co_variate_raster)  # One shot encoding bands with discrete values
    logger.debug(encoded_co_variate_raster.shape)

    modified_target_raster = handle_outliers(target_raster)
    logger.debug(modified_target_raster.shape)

    x, y = extract_data_from_co_variate_target(encoded_co_variate_raster, modified_target_raster)

    # Split the dataset into training, validation, and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.9, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.7, random_state=42)

    return {TRAIN_DATA: (x_train, y_train),
            TEST_DATA: (x_test, y_test),
            VALIDATION_DATA: (x_val, y_val)}


def process_raster_CNN():
    covariate_rast_path = "CO_VARIATE_RASTER"
    target_rast_path = "TGT_RASTER"

    path = Base.BASE_RASTER_PATH
    raster_abs_path = os.path.join(path, covariate_rast_path)
    target_abs_path = os.path.join(path, target_rast_path)

    data_set = load_process_data(raster_abs_path, target_abs_path)
    path = os.path.join(Base.LINEAR_TRAINING_DATA_PATH)

    store_data_files(data_set, path)


