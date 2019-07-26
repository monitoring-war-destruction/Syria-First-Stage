from datetime import date
import pandas as pd
import numpy as np
from tqdm import tqdm
from shapely.geometry import MultiPolygon, Point

from damage.features.base import Feature


class RasterSplitter(Feature):

    def __init__(self, patch_size, stride):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride

    def transform(self, data):
        raster_data = self._split_raster_data(data)
        return raster_data.set_index(['city', 'patch_id', 'date'])

    def _split_raster_data(self, data):
        rasters = [(key, value) for key, value in data.items() if 'raster' in key]
        tiles = []
        for name, raster in rasters:
            array = self._raster_to_array(raster)
            city, year, month, day = self.parse_raster_filename(name)
            polygons = self._get_polygons_from_data_dict_single_city(data, city)
            no_analysis_areas_polygon, populated_areas_polygon = polygons
            for w in tqdm(range(self.patch_size//2, (raster.width - self.patch_size//2), self.stride)):
                for h in range(self.patch_size//2, (raster.height - self.patch_size//2), self.stride):
                    longitude, latitude = raster.xy(h, w)
                    point_is_valid = self._is_point_valid(Point(longitude, latitude), populated_areas_polygon,
                                                          no_analysis_areas_polygon)
                    if not point_is_valid:
                        continue

                    left, right, top, bottom = self._get_tile_boundaries(w, h)
                    tile = {
                        'image': array[top:bottom, left:right],
                        'longitude': longitude,
                        'latitude': latitude,
                        'is_in_no_analysis?': no_analysis_areas_polygon.contains(Point(longitude, latitude)),
                        'city': city,
                        'date': date(year=year, month=month, day=day),
                        'patch_id': '{}-{}'.format(w, h),
                    }
                    tiles.append(tile)

        tiles = pd.DataFrame(tiles)
        return tiles

    @staticmethod
    def _get_polygons_from_data_dict_single_city(data_dict, city):
        # No analysis
        no_analysis_key = [key for key in data_dict if 'no_analysis' in key and city in key.lower()][0]
        no_analysis_areas = data_dict[no_analysis_key]
        no_analysis_areas_geometry = no_analysis_areas['geometry'].tolist()
        no_analysis_areas_polygon = MultiPolygon(no_analysis_areas_geometry).buffer(0)
        # Populated areas
        populated_areas_key = [key for key in data_dict if 'populated' in key][0]
        populated_areas = data_dict[populated_areas_key]
        populated_areas_city = populated_areas.loc[populated_areas['NAME_EN'].str.lower() == city]
        populated_areas_geometry = populated_areas_city['geometry'].tolist()
        populated_areas_polygon = MultiPolygon(populated_areas_geometry)
        return no_analysis_areas_polygon, populated_areas_polygon

    @staticmethod
    def _is_point_valid(point, populated_areas, no_analysis_areas):
        if not populated_areas.contains(point):
            return False

        elif no_analysis_areas.contains(point):
            return True

        else:
            return True

    def _get_tile_boundaries(self, w, h):
        left = (w - self.patch_size//2)
        right = (w + self.patch_size//2)
        top = (h - self.patch_size//2)
        bottom = (h + self.patch_size//2)
        return left, right, top, bottom

    @staticmethod
    def parse_raster_filename(filename):
        """This method assumes the following format:
        'raster_city_year_month_day...'
        """
        filename_split = filename.split('_')
        city = filename_split[1]
        year = int(filename_split[2])
        month = int(filename_split[3])
        day = int(filename_split[4])
        return city, year, month, day

    @staticmethod
    def _raster_to_array(raster):
        raster_array = raster.read(indexes=[1,2,3])
        raster_array = np.swapaxes(np.swapaxes(raster_array, 1, 2), 0, 2)
        return raster_array
