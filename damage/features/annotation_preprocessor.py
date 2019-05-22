import pandas as pd

from damage.utils import geo_location_index
from damage.features.base import Preprocessor


class AnnotationPreprocessor(Preprocessor):

    def __init__(self, grid_size=0.035):
        super().__init__()
        self.grid_size = grid_size

    def transform(self, data):
        for key, value in data.items():
            if 'annotation' not in key:
                continue
		
            value = self._add_latitude_and_longitude(value)
            value['location_index'] = geo_location_index(value['latitude'], value['longitude'], self.grid_size)
            value = value.rename({'StlmtNme': 'city'}, axis=1)
            value['city'] = value['city'].str.lower()
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
    def _crop_annotation_to_longitude_and_latitude(annotation_data, longitude, latitude):
        mask_longitude = annotation_data['longitude'] >= longitude[0]\
            & annotation_data['longitude'] < longitude[1]
        mask_latitude = annotation_data['latitude'] >= latitude[0]\
            & annotation_data['latitude'] < latitude[1]
        return annotation_data.loc[mask_longitude & mask_latitude]
