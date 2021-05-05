import os
from math import ceil
from time import time
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold

from damage.models import CNN, RandomSearch
from damage.data import DataStream, load_experiment_results
from damage import features
from damage.constants import PREDICTIONS_PATH, FEATURES_PATH, MODELS_PATH


parser = argparse.ArgumentParser()
parser.add_argument('--features', required=True)
parser.add_argument('--gpu')
parser.add_argument('--experiment')
args = vars(parser.parse_args())

os.environ['CUDA_VISIBLE_DEVICES'] = args.get('gpu', None) or '5'
features_file_name = args.get('features')

# Reading
features = pd.read_pickle('{}/{}'.format(FEATURES_PATH, features_file_name))\
    .dropna(subset=['image'])
# features_destroyed = features.loc[features['destroyed'] == 1].sample(500)
# features_non_destroyed = features.loc[features['destroyed'] == 0].sample(5000)
# features = pd.concat([features_destroyed, features_non_destroyed])
#  Modelling
Model = CNN

#  Choose Model
model = load_model('{}/model_{}.h5'.format(MODELS_PATH, args['experiment']))
experiments = load_experiment_results()
space = experiments.loc[
    experiments['name'] == 'experiment_{}.json'.format(args['experiment']),
    'space'
].iloc[0]
test_generator = DataStream._get_index_generator(features, space['batch_size'], KFold)
num_batches_test = len(test_generator)
test_generator = DataStream.get_test_data_generator_from_index(features['image'], test_generator)
test_dataset = Dataset.from_generator(lambda: test_generator, tf.float32)

#  Predict
print('Generating predictions')
predictions = model.predict_generator(test_dataset, steps=num_batches_test)
predictions = pd.DataFrame({
    'prediction': predictions.reshape(-1),
}, index=features.index)
file_name = '{}/prediction_{}.p'.format(PREDICTIONS_PATH, round(time()))
predictions.to_pickle(file_name)
print('Store predictions on file: {}'.format(file_name))
