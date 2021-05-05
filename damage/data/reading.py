import os
import json
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio

from damage.data.data_sources import DATA_SOURCES
from damage.constants import (
    RASTERS_PATH, ANNOTATIONS_PATH, POLYGONS_PATH, EXPERIMENTS_PATH, FEATURES_PATH
)
# Rasters = images (satellite)
# Annotations = labels UN
# Polygons = shapes to distinguish no-analysis area

def load_data_multiple_cities(cities, rasters_path=RASTERS_PATH, annotations_path=ANNOTATIONS_PATH,
                              polygons_path=POLYGONS_PATH):
    data = {}
    for city in cities:
        data = {**data, **load_data_single_city(city, rasters_path, annotations_path, polygons_path)}

    populated_areas = read_populated_areas(file_names=[
        '{}/populated_areas.shp'.format(polygons_path),
    ])
    data = {**populated_areas, **data}
    return data


def load_data_single_city(city, rasters_path, annotations_path, polygons_path):
    annotation_files = ['{}/{}'.format(annotations_path, f) for f in DATA_SOURCES[city]['annotations']]
    annotation_data = read_annotations(file_names=annotation_files)

    raster_files = ['{}/{}'.format(rasters_path, f) for f in DATA_SOURCES[city]['rasters']]
    raster_data = read_rasters(file_names=raster_files)

    no_analysis_files = ['{}/{}'.format(polygons_path, f) for f in DATA_SOURCES[city]['no_analysis']]
    no_analysis_area = read_no_analysis_areas(file_names=no_analysis_files)

    data = {**annotation_data, **raster_data, **no_analysis_area}
    return data


def read_annotations(file_names):
    data = {}
    for file_name in file_names:
        data['annotation_'+file_name.split('/')[-1]] = gpd.read_file(file_name)

    return data


def read_rasters(file_names):
    data = {}
    for file_name in file_names:
        data['raster_'+file_name.split('/')[-1]] = rasterio.open(file_name)

    return data


def read_populated_areas(file_names):
    data = {}
    for file_name in file_names:
        data['populated_areas_'+file_name.split('/')[-1]] = gpd.read_file(file_name)

    return data

def read_no_analysis_areas(file_names):
    data = {}
    for file_name in file_names:
        data['no_analysis_areas_'+file_name.split('/')[-1]] = gpd.read_file(file_name)

    return data

def load_experiment_results(path=EXPERIMENTS_PATH):
    experiment_files = os.listdir(path)
    experiment_results = []
    for file_name in experiment_files:
        with open('{}/{}'.format(path, file_name), 'r') as f:
            try:
                result = eval(json.load(f))
                result['id'] = int(file_name.split('_')[1].split('.')[0])
                result['name'] = file_name
                experiment_results.append(result)
            except:
                continue

    experiment_results = pd.DataFrame(experiment_results)
    experiment_results['val_precision_positives_last_epoch'] = experiment_results['val_precision_positives']\
        .apply(lambda x: np.nan if isinstance(x, float) else x[-1])
    experiment_results['val_recall_positives_last_epoch'] = experiment_results['val_recall_positives']\
        .apply(lambda x: np.nan if isinstance(x, float) else x[-1])
    return experiment_results


def load_features_multiple_cities(cities, test=False, sample=1):
    appended_features = []
    for city_feature_filename in cities:
        # Reading
        features_city = pd.read_pickle('{}/{}.p'.format(FEATURES_PATH, city_feature_filename))\
            .dropna(subset=['destroyed', 'image'])
        features_city = features_city.loc[features_city['no_analysis'] == 0]
        if test:
            features_destroyed = features_city.loc[features_city['destroyed'] == 1]\
                .sample(20, replace=True)
            features_non_destroyed = features_city.loc[features_city['destroyed'] == 0]\
                .sample(2000, replace=True)
            features_city = pd.concat([features_destroyed, features_non_destroyed])
        elif sample < 1:
            features_destroyed = features_city.loc[features_city['destroyed'] == 1]
            features_destroyed = features_destroyed.sample(
                int(len(features_destroyed)*sample)
            )
            features_non_destroyed = features_city.loc[features_city['destroyed'] == 0]
            features_destroyed = features_non_destroyed.sample(
                int(len(features_non_destroyed)*sample)
            )
            features_city = pd.concat([features_destroyed, features_non_destroyed])

        appended_features.append(features_city)

    features = pd.concat(appended_features)
    return features
