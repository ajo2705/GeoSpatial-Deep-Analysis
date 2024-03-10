import numpy as np
import rasterio
import torch
import logging

from abc import ABC, abstractmethod
from math import floor, ceil

from raster_util import detect_outliers_iqr, plot_data_non_outliers, BoundaryType, percentile_based_boundary

logger = logging.getLogger('cnn_logger')


class RasterConverter(ABC):
    def __init__(self, co_variate_raster_path, target_raster_path, patch_size):
        self.co_variate_raster, self.target_raster = RasterConverter.load_raster_data(co_variate_raster_path,
                                                                                      target_raster_path)

        self.patch_size = patch_size

    @staticmethod
    def load_raster_data(co_variate_raster_path, target_raster_path):
        """
        Read the source raster with X values and target raster with Y values.
        The co-variate raster contains all features combined in a single raster as different bands.
        :param co_variate_raster_path:
        :param target_raster_path:
        :return:
        """
        with rasterio.open(co_variate_raster_path) as src:
            co_variate_raster = src.read()
            no_data_band_index = [idx for idx in range(1, src.count + 1) if src.tags(idx)['NAME'] == 'NoData_Mask'][0]

        with rasterio.open(target_raster_path) as src:
            target_raster = src.read(1)

            # Mask the NaN values in the target raster using the NoData mask from the co-variate raster
            nodata_mask = co_variate_raster[no_data_band_index - 1].astype(bool)
            target_raster[nodata_mask] = np.nan

        return co_variate_raster, target_raster

    @staticmethod
    def remove_outliers(target_raster):
        # Remove NaN values from the target raster to calculate outlier indices
        non_nan_target_values = target_raster[~np.isnan(target_raster)]
        outlier_indices = detect_outliers_iqr(non_nan_target_values, q1=0.25, q3=0.75)

        # Calculate the soil carbon class boundaries based on the percentiles
        non_outlier_target_values = np.delete(non_nan_target_values, outlier_indices)
        return non_outlier_target_values

    def plot_hist_data(self, bins=50):
        non_outlier_targets = self.remove_outliers(self.target_raster)
        plot_data_non_outliers(non_outlier_targets, bins)

    def filter_datapoints(self):
        """
        Take relevant points from sparse target raster and creates a bounding patch of specified pixels from
        the co-variate raster. If nan value in co-variate raster in the bounding patch, the point is ignored.
        We are taking such an approach as the target raster is sparse.
        :return:
        """

        logger.debug(f"Shape of co-variate raster: {self.co_variate_raster.shape}")

        non_outlier_target_values = self.remove_outliers(self.target_raster)
        min_target_val = floor(np.min(non_outlier_target_values))
        max_target_val = ceil(np.max(non_outlier_target_values)) + 1

        image_patches = []
        class_labels = []
        height, width = self.target_raster.shape
        half_patch_size = self.patch_size // 2
        for y in range(half_patch_size, height - half_patch_size):
            for x in range(half_patch_size, width - half_patch_size):
                # Get target value (Y vals)
                target_value = self.target_raster[y, x]
                self.__filter_target_nan(class_labels, half_patch_size, image_patches, min_target_val, max_target_val,
                                         target_value, x, y)
        return class_labels, image_patches

    def __filter_target_nan(self, class_labels, half_patch_size, image_patches, min_target_val, max_target_val,
                            target_value, x, y):
        """
        Creating bounding patch corresponding to target raster point
        """
        if not np.isnan(target_value) and min_target_val <= target_value < max_target_val:
            patch = self.co_variate_raster[:-1, y - half_patch_size:y + half_patch_size, x - half_patch_size:
                                                                                             x + half_patch_size]

            # Check for NaN values in the patch and skip if any are found
            self._filter_patch_nan(class_labels, image_patches, patch, target_value)

    @abstractmethod
    def _filter_patch_nan(self, class_labels, image_patches, patch, target_value):
        """
        Implementation in subclasses
        :param class_labels:
        :param image_patches:
        :param patch:
        :param target_value:
        :return:
        """
        pass


class ClassifierRasterConverter(RasterConverter):
    """
    Raster processing for performing classification. Target raster processing will be tuned for generating
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_boundaries = None

    def generate_class_boundaries(self, boundary_type, custom_boundary=[]):
        non_outlier_targets = self.remove_outliers(self.target_raster)
        if boundary_type == BoundaryType.Percentile:
            self.class_boundaries = percentile_based_boundary(non_outlier_targets)
        else:
            self.class_boundaries = custom_boundary
            self.class_boundaries.append(ceil(np.max(non_outlier_targets) + 1))
            self.class_boundaries.insert(0, floor(np.min(non_outlier_targets)))
        return self.class_boundaries

    def _filter_patch_nan(self, class_labels, image_patches, patch, target_value):
        if np.isnan(patch).any() or -9999.0 in patch:
            return
        image_patches.append(patch)
        soil_carbon_class = np.digitize(target_value, self.class_boundaries) - 1
        class_labels.append(soil_carbon_class)


class LinearRasterConverter(RasterConverter):
    """
    Raster converter for linear regression operations
    """
    FILTER_CHANNELS = ["soil_taxonomy_refined", "LC_type1", "CDL_1km", 'elev', 'srad', 'slope', 'local_upslope_curvature',
                       'local_downslope_curvature', 'aspect', 'bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_15',
                       'NDVI_p50', 'EVI_p0', 'Final_sm', 'clay_b0', 'clay_b10', 'ai', 'et0', 'SBIO_7', '5to15cm_SBIO_3',
                       '5to15cm_SBIO_4', '5to15cm_SBIO_5', 'NoData_Mask']

    def __init__(self, co_variate_raster_path, *args, **kwargs):
        super().__init__(co_variate_raster_path, *args, **kwargs)
        self.class_datas, self.no_data_band_index = self.__get_class_datas(co_variate_raster_path)

    @staticmethod
    def __get_class_datas(co_variate_raster_path):
        """
        Get name of raster bands
        """
        with rasterio.open(co_variate_raster_path) as src:
            class_datas = [src.tags(idx)['NAME']for idx in range(1, src.count + 1)]
            no_data_band_index = [idx for idx in range(1, src.count + 1) if src.tags(idx)['NAME'] == 'NoData_Mask'][0]

        return class_datas, no_data_band_index

    def filter_channels(self):
        """
        Filter channels based on FILTER_CHANNELS
        """
        filter_channels = self.FILTER_CHANNELS
        index_list = [index for index, item in enumerate(self.class_datas) if item in filter_channels]

        self.class_datas = [self.class_datas[index] for index in index_list]
        logger.debug(f"Valid channels are : {self.class_datas}")

        self._filter_channels_based_on_index(index_list)

    def _filter_channels_based_on_index(self, index_list):
        total_channels = [self.co_variate_raster[i, :, :] for i in index_list]
        co_variate_raster = np.stack(total_channels, axis=0)

        logger.debug(f"Shape of filtered_raster :{co_variate_raster.shape}")

        nodata_mask = self.co_variate_raster[self.no_data_band_index - 1]
        assert np.array_equal(nodata_mask, co_variate_raster[-1, :, :])

        self.co_variate_raster = co_variate_raster

        # New no_data_band index position is the last position
        # Change according to need --> hardcoded now
        self.no_data_band_index = len(self.class_datas) - 1

    def normalise_classification_channels(self):
        """
        One hot encode of classification channels.
        Channels having value greater than 1 are classification channels here.
        """
        total_channels = []
        normal_channels = []
        for i in range(len(self.class_datas)):
            channel = self.co_variate_raster[i, :, :]
            max_val = np.nanmax(channel)

            if max_val > 1.0:
                distinct_values = np.unique(channel[~np.isnan(channel)])
                num_classes = len(distinct_values)
                value_to_index = {value: index for index, value in enumerate(distinct_values)}

                combined_array = np.zeros((num_classes,) + channel.shape)

                for i in range(channel.shape[0]):
                    for j in range(channel.shape[1]):
                        value = channel[i, j]
                        if not np.isnan(value):
                            index = value_to_index[value]
                            combined_array[index, i, j] = 1
                total_channels.append(combined_array)
            else:
                normal_channels.append(channel)

        co_variate_raster_1 = np.stack(normal_channels, axis=0)
        co_variate_raster_2 = np.concatenate(total_channels, axis=0)

        combined_raster = np.concatenate((co_variate_raster_2, co_variate_raster_1), axis=0)
        logger.debug("Normalized raster shapes")
        logger.debug(co_variate_raster_1.shape)
        logger.debug(co_variate_raster_2.shape)
        logger.debug(f"combined raster shape: {combined_raster.shape}")

        nodata_mask = self.co_variate_raster[self.no_data_band_index]
        assert np.array_equal(nodata_mask, combined_raster[-1,:, :])

        self.co_variate_raster = combined_raster

    def _filter_patch_nan(self, targets, image_patches, patch, target_value):
        if np.isnan(patch).any() or -9999.0 in patch:
            return
        image_patches.append(patch)
        targets.append(target_value)
