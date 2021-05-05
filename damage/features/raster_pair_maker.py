import pandas as pd
import numpy as np

from damage.features.base import Feature


class RasterPairMaker(Feature):

    """ This class makes pairs of raster images in different points in time
    by concatenating the first image to every posterior one in each city
    (going from 3 to 6 channels).
    """

    def transform(self, data):
        raster_pairs = self._make_raster_pairs_all_cities(data['RasterSplitter'].reset_index())
        data.pop('RasterSplitter') # Hack to avoid memory problems
        # We drop nans on date because those are the images that come before
        # any annotation, and cannot be used for training
        return raster_pairs.dropna(subset=['date']).set_index(['city', 'patch_id', 'date'])

    def _make_raster_pairs_all_cities(self, raster_data):
        raster_pairs_all_cities = []
        for city in raster_data['city'].unique():
            raster_data_single_city = raster_data.loc[raster_data['city'] == city]
            raster_pairs_single_city = self._make_raster_pairs_single_city(raster_data_single_city)
            del raster_data_single_city
            raster_pairs_all_cities.append(raster_pairs_single_city)

        del raster_data
        raster_pairs_all_cities = pd.concat(raster_pairs_all_cities)
        return raster_pairs_all_cities

    def _make_raster_pairs_single_city(self, raster_data):
        """ This method creates pairs of images
        (image_0, image_1), (image_0, image_2) by concatenating
        on the third axis (3 to 6 channel images) for a single city.
        """
        dates = raster_data['date'].unique()
        print('**** raster_pair_marker.py _make_raster_pairs_single_city dates={}'.format(dates))
        first_date = dates.min()
        first_raster = raster_data.loc[raster_data['date'] == first_date]
        combined_rasters = []
        for date in dates:
            if date == first_date:
                continue

            single_raster = raster_data.loc[raster_data['date'] == date]
            combined_raster = self._make_single_raster_pair_dataframe(first_raster, single_raster)
            del single_raster
            combined_rasters.append(combined_raster)

        combined_rasters = pd.concat(combined_rasters)
        del first_raster
        del raster_data
        return combined_rasters

    def _make_single_raster_pair_dataframe(self, common_raster, variable_raster):
        """ This method takes two rasters and combined them
        by concatenating the images in them on the second axis.
        It keeps the variable_raster as the data structure and
        concatenates the images of the common_raster to the ones
        in the variable_raster.
        """
        combined_raster = pd.merge(variable_raster,
                                   common_raster[['patch_id', 'image']],
                                   on='patch_id')
        assert len(combined_raster) == len(variable_raster)
        del variable_raster
        combined_raster['image']  = combined_raster.apply(
            lambda row: np.concatenate([row['image_x'], row['image_y']], axis=2), axis=1)
        combined_raster = combined_raster.drop(['image_x', 'image_y'], axis=1)
        return combined_raster

    def _make_single_raster_pair_list(self, common_raster, variable_raster):
        """ This method takes two rasters and combined them
        by concatenating the images in them on the second axis.
        It keeps the variable_raster as the data structure and
        concatenates the images of the common_raster to the ones
        in the variable_raster.
        """
        combined_raster = pd.merge(variable_raster,
                                   common_raster[['patch_id', 'image']],
                                   on='patch_id')
        assert len(combined_raster) == len(variable_raster)
        del variable_raster
        combined_images = []
        for index, row in combined_raster.iterrows():
            combined_images.append(np.concatenate([row['image_x'], row['image_y']], axis=2))

        return combined_images
