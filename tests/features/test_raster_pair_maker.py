import unittest
from datetime import date
import pandas as pd
import numpy as np

from damage.features import RasterPairMaker
from tests.utils import assertFrameEqual


class TestRasterPairMaker(unittest.TestCase):

    def setUp(self):
        self.raster_pair_maker = RasterPairMaker()

    def test_make_single_raster_pair_expected_input_returns_expected_output(self):
        first_raster = pd.DataFrame({
            'city': ['a'],
            'patch_id': [0],
            'image': [np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])],
            'date': [date(2018, 1, 1)],
        })
        posterior_raster = pd.DataFrame({
            'city': ['a'],
            'patch_id': [0],
            'image': [np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])],
            'raster_date': [date(2018, 1, 1)],
            'date': [date(2018, 1, 1)],
        })
        raster_pair = self.raster_pair_maker._make_single_raster_pair_dataframe(first_raster, posterior_raster)
        expected_output = pd.DataFrame({
            'city': ['a'],
            'patch_id': [0],
            'image': [np.array([[[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]]])],
            'raster_date': [date(2018, 1, 1)],
            'date': [date(2018, 1, 1)],
        })
        assertFrameEqual(raster_pair, expected_output)

    def test_make_raster_pairs_single_city_expected_input_returns_expected_output(self):
        raster_data = pd.DataFrame({
            'city': ['a', 'a', 'a'],
            'patch_id': [0, 0, 0],
            'image': [np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
                      np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]),
                      np.array([[[2, 2, 2], [2, 2, 2], [2, 2, 2]]])],
            'raster_date': [date(2017, 1, 1), date(2018, 1, 1), date(2019, 1, 1)],
            'date': [date(2017, 1, 1), date(2018, 1, 1), date(2019, 1, 1)],
        })
        output = self.raster_pair_maker._make_raster_pairs_single_city(raster_data)
        expected_output = pd.DataFrame({
            'city': ['a', 'a'],
            'patch_id': [0, 0],
            'image': [
                np.array([[[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]]]),
                np.array([[[2, 2, 2, 0, 0, 0], [2, 2, 2, 0, 0, 0], [2, 2, 2, 0, 0, 0]]]),
            ],
            'raster_date': [date(2018, 1, 1), date(2019, 1, 1)],
            'date': [date(2018, 1, 1), date(2019, 1, 1)],
        })
        assertFrameEqual(output, expected_output)

    def test_make_raster_pairs_all_cities_expected_input_returns_expected_output(self):
        raster_data = pd.DataFrame({
            'patch_id': [0, 0],
            'image': [np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]),
                      np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])],
            'date': [date(2017, 1, 1), date(2018, 1, 1)],
            'raster_date': [date(2017, 1, 1), date(2018, 1, 1)],
            'city': ['a', 'a'],
        })
        output = self.raster_pair_maker._make_raster_pairs_all_cities(raster_data)
        expected_output = pd.DataFrame({
            'patch_id': [0],
            'image': [np.array([[[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]]])],
            'date': [date(2018, 1, 1)],
            'raster_date': [date(2018, 1, 1)],
            'city': ['a'],
        })
        assertFrameEqual(output, expected_output)
