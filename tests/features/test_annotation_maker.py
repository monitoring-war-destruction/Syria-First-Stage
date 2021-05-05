import unittest
from datetime import date, timedelta
import pandas as pd
import numpy as np

from damage.features import AnnotationMaker
from tests.utils import assertFrameEqual


class TestAnnotationMaker(unittest.TestCase):

    def setUp(self):
        patch_size = 64
        time_to_annotation_threshold = timedelta(weeks=1)
        self.annotation_maker = AnnotationMaker(patch_size, time_to_annotation_threshold)

    def test_combine_annotation_data(self):
        city_a_data = pd.DataFrame({
            'city': ['a'],
            'dmg': ['Destroyed'],
        })
        city_b_data = pd.DataFrame({
            'city': ['b'],
            'dmg': ['Severe Damage'],
        })
        annotation_data = {
            'annotation_city_a': city_a_data,
            'annotation_city_b': city_b_data,
        }
        expected_output = pd.DataFrame({
            'city': ['a', 'b'],
            'dmg': ['Destroyed', 'Severe Damage'],
        })
        output = self.annotation_maker._combine_annotation_data(annotation_data)
        assertFrameEqual(output, expected_output)

    def test_combine_annotations_and_rasters(self):
        annotation_data = pd.DataFrame({
            'city': ['a', 'a'],
            'patch_id': ['1', '2'],
            'date': [
                date(2017, 5, 20),
                date(2017, 5, 20),
            ],
            'damage_num': [1, 3],
        }).set_index(['city', 'patch_id', 'date'])
        raster_data = pd.DataFrame({
            'city': ['a'] * 9,
            'patch_id': ['1', '2', '3', '1', '2', '3', '1', '2', '3'],
            'date': [
                date(2016, 5, 20),
                date(2016, 5, 20),
                date(2016, 5, 20),
                date(2017, 5, 20),
                date(2017, 5, 20),
                date(2017, 5, 20),
                date(2018, 5, 20),
                date(2018, 5, 20),
                date(2018, 5, 20),
            ],
            'image': [f'image_{i}' for i in range(9)],
        }).set_index(['city', 'patch_id', 'date'])
        expected_output = pd.DataFrame({
            'city': ['a'] * 9,
            'patch_id': ['1', '2', '3', '1', '2', '3', '1', '2', '3'],
            'date': [
                date(2016, 5, 20),
                date(2016, 5, 20),
                date(2016, 5, 20),
                date(2017, 5, 20),
                date(2017, 5, 20),
                date(2017, 5, 20),
                date(2018, 5, 20),
                date(2018, 5, 20),
                date(2018, 5, 20),
            ],
            'annotation_date': [
                np.nan,
                np.nan,
                np.nan,
                date(2017, 5, 20),
                date(2017, 5, 20),
                date(2017, 5, 20),
                np.nan,
                np.nan,
                np.nan,
            ],
            'destroyed': [0, np.nan, 0, 0, 1, 0, np.nan, 1, np.nan],
            'image': [f'image_{i}' for i in range(9)],
        }).sort_values(['city', 'patch_id', 'date'])
        expected_output['annotation_date'] = pd.to_datetime(expected_output['annotation_date'])
        expected_output['date'] = pd.to_datetime(expected_output['date'])
        output = self.annotation_maker._combine_annotations_and_rasters(
            annotation_data,
            raster_data
        )
        assertFrameEqual(output, expected_output)

    '''
    def test_is_date_previous_or_same_to_date_previous_date(self):
        date_0, date_1 = date(2016, 1, 1), date(2016, 1, 2)
        output = self.annotation_maker._is_date_previous_or_same_to_date(date_0, date_1)
        self.assertTrue(output)

    def test_is_date_previous_or_same_to_date_posterior_date(self):
        date_0, date_1 = date(2016, 1, 3), date(2016, 1, 2)
        output = self.annotation_maker._is_date_previous_or_same_to_date(date_0, date_1)
        self.assertFalse(output)

    def test_get_closest_previous_data(self):
        pool_of_dates = [date(2016, 1, 1), date(2017, 1, 1), date(2018, 1, 1)]
        single_date = date(2016, 12, 31)
        output = self.annotation_maker._get_closest_previous_date(single_date, pool_of_dates)
        expected_output = date(2016, 1, 1)
        self.assertEqual(expected_output, output)

    def test_get_raster_dates_with_annotation_data(self):
        raster_dates = pd.DataFrame({
            'city': ['a'] * 3,
            'raster_date': [date(2015, 1, 1), date(2016, 2, 1), date(2017, 9, 1)],
            'annotation_date': [None, date(2016, 1, 1), date(2017, 1, 1)],
        })
        annotation_data = pd.DataFrame({
            'date': [date(2016, 1, 1), date(2017, 1, 1), date(2017, 10, 1), date(2018, 1, 1)],
            'city': ['a'] * 4,
            'damage': [1] * 4,
        })
        output = self.annotation_maker._get_raster_dates_with_annotation_data(raster_dates, annotation_data)
        expected_output = pd.DataFrame({
            'city': ['a'] * 2,
            'date': [date(2016, 2, 1), date(2017, 9, 1)],
            'annotation_date': [date(2016, 1, 1), date(2017, 1, 1)],
            'damage': [1, 1],
        })
        assertFrameEqual(output, expected_output)
    
    def test_get_long_and_short_gap_annotations(self):
        annotation_data = pd.DataFrame({
            'city': ['a'] * 2,
            'date': [date(2016, 1, 5), date(2017, 9, 1)],
            'annotation_date': [date(2016, 1, 1), date(2017, 1, 1)],
            'damage': [1, 1],
        })
        output = self.annotation_maker._get_long_and_short_gap_annotations(annotation_data)
        expected_output_0 = pd.DataFrame({
            'city': ['a'],
            'date': [date(2017, 9, 1)],
            'annotation_date': [date(2017, 1, 1)],
            'damage': [1],
        })
        expected_output_1 = pd.DataFrame({
            'city': ['a'],
            'date': [date(2016, 1, 5)],
            'annotation_date': [date(2016, 1, 1)],
            'damage': [1],
        })
        assertFrameEqual(output[0], expected_output_0)
        assertFrameEqual(output[1], expected_output_1)
    
    def test_combine_long_and_short_gap_annotations_with_raster_data_happy_path(self):
        short_gap = pd.DataFrame({
            'city': ['a']*2,
            'date': [date(2016, 1, 5), date(2016, 1, 5)],
            'annotation_date': [date(2016, 1, 1), date(2016, 1, 1)],
            'patch_id': ['a', 'b'],
            'damage_num': [1, 1],
        })
        long_gap = pd.DataFrame({
            'city': ['a']*2,
            'date': [date(2017, 9, 1), date(2017, 9, 1)],
            'annotation_date': [date(2017, 1, 1), date(2017, 1, 1)],
            'patch_id': ['a', 'b'],
            'damage_num': [1, 3],
        })
        raster_data = pd.DataFrame({
            'city': ['a']*4,
            'date': [date(2016, 1, 5), date(2016, 1, 5), date(2017, 9, 1), date(2017, 9, 1)],
            'patch_id': ['a', 'b', 'a', 'b'],
        })
        expected_output = pd.DataFrame({
            'city': ['a']*4,
            'date': [date(2016, 1, 5), date(2016, 1, 5), date(2017, 9, 1), date(2017, 9, 1)],
            'annotation_date': [date(2016, 1, 1), date(2016, 1, 1),
                                date(2017, 1, 1), date(2017, 1, 1)],
            'patch_id': ['a', 'b', 'a', 'b'],
            'damage_num': [1., 1., np.nan, 3.],
        }).sort_values('patch_id')
        self.annotation_maker.last_annotation_date = date(2017, 1, 1)
        output = self.annotation_maker._combine_long_and_short_gap_annotations_with_raster_data(
            long_gap,
            short_gap,
            raster_data
        )
        assertFrameEqual(output, expected_output)

    def test_combine_long_and_short_gap_annotations_with_raster_data_case_0(self):
        short_gap = pd.DataFrame({
            'city': ['a']*2,
            'date': [date(2016, 1, 5), date(2016, 1, 5)],
            'annotation_date': [date(2016, 1, 1), date(2016, 1, 1)],
            'patch_id': ['a', 'b'],
            'damage_num': [0, 3],
        })
        long_gap = pd.DataFrame({
            'city': ['a']*4,
            'date': [date(2015, 1, 5), date(2015, 1, 5),
                     date(2017, 1, 5), date(2017, 1, 5)],
            'annotation_date': [np.nan, np.nan,
                                date(2016, 1, 1), date(2016, 1, 1)]
            'patch_id': ['a', 'a', 'b', 'b'],
            'damage_num': [1, 3],
        })
        raster_data = pd.DataFrame({
            'city': ['a']*4,
            'date': [date(2016, 1, 5), date(2016, 1, 5), date(2017, 9, 1), date(2017, 9, 1)],
            'patch_id': ['a', 'b', 'a', 'b'],
        })
        expected_output = pd.DataFrame({
            'city': ['a']*4,
            'date': [date(2016, 1, 5), date(2016, 1, 5), date(2017, 9, 1), date(2017, 9, 1)],
            'annotation_date': [date(2016, 1, 1), date(2016, 1, 1),
                                date(2017, 1, 1), date(2017, 1, 1)],
            'patch_id': ['a', 'b', 'a', 'b'],
            'damage_num': [1., 1., np.nan, 3.],
        }).sort_values('patch_id')
        self.annotation_maker.last_annotation_date = date(2017, 1, 1)
        output = self.annotation_maker._combine_long_and_short_gap_annotations_with_raster_data(
            long_gap,
            short_gap,
            raster_data
        )
        assertFrameEqual(output, expected_output)
    '''
