import os
import random
import argparse
import json
from time import time
import pandas as pd
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.errors import ResourceExhaustedError

from damage.data import DataStream
from damage.models import CNN, RandomSearch, CNNPreTrained


parser = argparse.ArgumentParser()
parser.add_argument('--features', required=True)
parser.add_argument('--gpu')
args = vars(parser.parse_args())


os.environ['CUDA_VISIBLE_DEVICES'] = args.get('gpu', None) or '5'
RESULTS_PATH = 'logs/experiments'
FEATURES_PATH = 'logs/features'
features_file_name = args.get('features')
list_feature_filenames_by_city = features_file_name.split(',')

appended_features = []
# choose some city as base city (e.g. aleppo)
base_city = 'aleppo.p'
base_city_ix = list_feature_filenames_by_city.index(base_city)
# base city needs to be the first in the list
list_feature_filenames_by_city[0], list_feature_filenames_by_city[base_city_ix]=
list_feature_filenames_by_city[base_city_ix], list_feature_filenames_by_city[0]

for city_feature_filename in list_feature_filenames_by_city:
    # Reading
    features_city = pd.read_pickle('{}/{}'.format(FEATURES_PATH, city_feature_filename)).dropna(subset=['destroyed'])
    appended_features.append(features_city)
    if city_feature_filename.lower() == base_city:
        n_base = features_city.shape[0]
    else:
        # upsampling for cities with less rasters
        n_df = features_city.shape[0]
        #sample with replacement
        ix = choices(np.arange(n_df), k=n_base-n_df)
        xtra_df = df_city.loc[ix, :]
        appended_dfs.append(xtra_df)
    #features_destroyed = features.loc[features['destroyed'] == 1].sample(200)
    #features_non_destroyed = features.loc[features['destroyed'] == 0].sample(2000)
    #features = pd.concat([features_destroyed, features_non_destroyed])
features = pd.concat(appended_features)

#### Modelling
sampler = RandomSearch()
models = {
    CNN: sampler.sample_cnn,
    CNNPreTrained: sampler.sample_cnn_pretrained,
}
Model = random.choice([CNN])
sample_func = models[Model]
spaces = sample_func(500)
# Do splits
class_proportion = {
    1: 0.3,
}
batch_size = spaces[0]['batch_size']
test_batch_size = 500
train_proportion = 0.7
data_stream = DataStream(batch_size=batch_size, train_proportion=train_proportion,
                         class_proportion=class_proportion, test_batch_size=test_batch_size)

unique_patches = features.index.get_level_values('patch_id').unique().tolist()
train_patches = random.sample(unique_patches, round(len(unique_patches)*train_proportion))
train_data = features.loc[features.index.get_level_values('patch_id').isin(train_patches)]
train_data_upsampled = data_stream._upsample_class_proportion(train_data, class_proportion).sample(frac=1)
test_patches = list(set(unique_patches) - set(train_patches))
test_data = features.loc[features.index.get_level_values('patch_id').isin(test_patches)]

train_indices = data_stream._get_index_generator(train_data_upsampled, batch_size)
test_indices = data_stream._get_index_generator(test_data, test_batch_size)
train_generator = data_stream.get_train_data_generator_from_index(
    [train_data_upsampled['image'], train_data_upsampled['destroyed']], train_indices)
test_generator = data_stream.get_train_data_generator_from_index(
    [test_data['image'], test_data['destroyed']], test_indices)
train_dataset = Dataset.from_generator(lambda: train_generator, (tf.float32, tf.int32))
test_dataset = Dataset.from_generator(lambda: test_generator, (tf.float32, tf.int32))

num_batches = len(train_indices)
num_batches_test = len(test_indices)
#Validate
for space in spaces:
    space['batch_size'] = batch_size
    space['class_weight'] = {
        0: (class_proportion[1] * space['class_weight']),
        1: 1 - (class_proportion[1] * space['class_weight']),
    }
    print('Now validating:\n')
    print(space)
    try:
        model = Model(**space)
        losses = model.validate_generator(train_dataset, test_dataset,
                                          steps_per_epoch=num_batches,
                                          validation_steps=num_batches_test,
                                          **space)
    except Exception as e:
        losses = {'log': str(e)}

    losses['model'] = str(Model)
    losses['space'] = space
    losses['features'] = features_file_name
    losses['num_batches_train'] = num_batches
    losses['num_batches_test'] = num_batches_test
    identifier = round(time())
    with open('{}/experiment_{}.json'.format(RESULTS_PATH, identifier), 'w') as f:
        json.dump(str(losses), f)
    if 'val_recall_positives' in losses.keys():
        if losses['val_recall_positives'][-1] > 0.4 and losses['val_precision_positives'][-1] > 0.1:
            model.save('logs/models/model_{}.h5'.format(identifier))
