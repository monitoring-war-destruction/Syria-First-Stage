import pandas as pd

from damage.utils import geo_location_index
from damage.features.base import Preprocessor


class AnnotationPreprocessor(Preprocessor):

    def transform(self, data):
        for key, value in data.items():
            if 'annotation' not in key:
                continue
            if 'Aleppo' in key:
                value = self.apply_aleppo_hacks(value)

            value = self._add_latitude_and_longitude(value)
            value = value.rename({'StlmtNme': 'city'}, axis=1)
            value['city'] = value['city'].str.lower().apply(lambda x: x.split(' ')[0])
            if 'Damascus' in key:
                value['city'] = 'damascus'
            elif 'Deir' in key:
                value['city'] = 'deir-ez-zor'

            assert value['city'].nunique() == 1
            city = value['city'].unique()[0]
            raster_key = [key for key in data.keys() if 'raster' in key and city in key][0]
            width, height = data[raster_key].width, data[raster_key].height
            value['row'], value['column'] = data[raster_key].index(value['longitude'], value['latitude'])
            value = self._crop_annotation_to_image_dimensions(value, {'height': height, 'width': width})
            value = self._unpivot_annotation(value)
            value['damage_num'] = self._get_damage_numerical(value['damage'])
            data[key] = value

        return data

    def _add_latitude_and_longitude(self, annotation_data):
        annotation_data['latitude'] = annotation_data['geometry'].apply(lambda point: point.y)
        annotation_data['longitude'] = annotation_data['geometry'].apply(lambda point: point.x)
        return annotation_data

    def _unpivot_annotation(self, annotation_data):
        damage_columns = [col for col in annotation_data if 'DmgCls' in col and 'Grp' not in col]
        other_columns = [col for col in annotation_data if col not in damage_columns]
        annotation_data = pd.melt(annotation_data, id_vars=other_columns, value_vars=damage_columns,
                                  value_name='damage')
        annotation_data['date'] = self._create_date_column(annotation_data)
        id_vars = ['latitude', 'longitude', 'date']
        # Some observations exist in one of the DmgCls columns but are None in the rest
        # (they did not assess them), we will drop those.
        annotation_data = annotation_data.dropna(subset=['date']).drop(['variable'], axis=1)
        return annotation_data

    @staticmethod
    def _create_date_column(annotation_data):
        date_column = annotation_data.apply(
            lambda x: x['SensDt_{}'.format(x['variable'][-1])] if '_' in x['variable'] else x['SensDt'], axis=1)
        date_column = pd.to_datetime(date_column).dt.date
        return date_column

    @staticmethod
    def _get_damage_numerical(damage_data):
        mapping = {'No Visible Damage': 0, 'Moderate Damage': 1, 'Severe Damage': 2, 'Destroyed': 3}
        return damage_data.map(mapping)

    @staticmethod
    def _crop_annotation_to_image_dimensions(annotation_data, dimensions):
        mask = (annotation_data['row'] < dimensions['height'])\
            & (annotation_data['row'] >= 0)\
            & (annotation_data['column'] < dimensions['width'])\
            & (annotation_data['column'] >= 0)
        return annotation_data.loc[mask]

    @staticmethod
    def apply_aleppo_hacks(annotation_data):
        """One annotation date has two dates. One of the dates
        (2015-05-01) only appears 11 times. If left as it is
        they will be considered different annotations and
        rasters will be matched with the date with few annotations.
        """
        annotation_data['SensDt_3'] = annotation_data['SensDt_3'].apply(lambda x: '2015-04-26' if x == '2015-05-01' else x)
        return annotation_data
