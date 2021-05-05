import os
import random
import argparse
import json
from time import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.errors import ResourceExhaustedError
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
from damage.data import DataStream, load_features_multiple_cities
from damage.models import CNN, RandomSearch, CNNPreTrained
from damage.constants import EXPERIMENTS_PATH, FEATURES_PATH, PREDICTIONS_PATH, MODELS_PATH


parser = argparse.ArgumentParser()
parser.add_argument('--features', required=True)
parser.add_argument('--gpu')
args = vars(parser.parse_args())

os.environ['CUDA_VISIBLE_DEVICES'] = args.get('gpu')
TEST_MODE = os.environ.get('SYRIA_TEST', False)
SAMPLE_CITY = 1
features_file_name = args.get('features')
cities = [c.split('.p')[0] for c in features_file_name.split(',')]

#### Loading features
features = load_features_multiple_cities(cities, TEST_MODE, SAMPLE_CITY)

#### Modelling
sampler = RandomSearch()
models = {
    CNN: sampler.sample_cnn,
    CNNPreTrained: sampler.sample_cnn_pretrained,
}
Model = random.choice([CNN])
sample_func = models[Model]
augment_brightness = random.choice([False])
augment_flip = random.choice([False])
class_proportion = {
    1: .3
}
spaces = sample_func(500)
batch_size = spaces[0]['batch_size']
test_batch_size = 200
train_proportion = 0.7
data_stream = DataStream(
    batch_size=batch_size,
    train_proportion=train_proportion,
    class_proportion=class_proportion,
    test_batch_size=test_batch_size
)
unique_patches = features.index.get_level_values('patch_id').unique().tolist()
train_patches = random.sample(unique_patches, round(len(unique_patches)*train_proportion))
train_data = features.loc[features.index.get_level_values('patch_id').isin(train_patches)]
train_data_upsampled = data_stream._upsample_class_proportion(train_data, class_proportion).sample(frac=1)
test_patches = list(set(unique_patches) - set(train_patches))
test_data = features.loc[features.index.get_level_values('patch_id').isin(test_patches)]

train_indices = data_stream._get_index_generator(train_data_upsampled, batch_size)
test_indices = data_stream._get_index_generator(test_data, test_batch_size)
train_generator = data_stream.get_train_data_generator_from_index(
    data=[train_data_upsampled['image'], train_data_upsampled['destroyed']],
    index=train_indices,
    augment_flip=augment_flip,
    augment_brightness=augment_brightness,
)
test_generator = data_stream.get_train_data_generator_from_index(
    data=[test_data['image'], test_data['destroyed']],
    index=test_indices,
    augment_flip=False,
    augment_brightness=False,
)
train_dataset = Dataset.from_generator(lambda: train_generator, (tf.float32, tf.int32))
test_dataset = Dataset.from_generator(lambda: test_generator, (tf.float32, tf.int32))

num_batches = len(train_indices)
num_batches_test = len(test_indices)
#Validate
for space in spaces:
    if TEST_MODE:
        space['epochs'] = 1
        space['convolutional_layers'] = space['convolutional_layers'][:1]
        space['dense_units'] = 16

    space['batch_size'] = batch_size
    space['prop_1_to_0'] = class_proportion[1]
    space['prop_1_train'] = train_data_upsampled['destroyed'].mean()
    class_weight_0 = max(class_proportion[1] * space['class_weight'], 0.1)
    class_weight_1 = min((1 - (class_proportion[1] * space['class_weight'])), 0.9)
    space['class_weight'] = {
        0: class_weight_0,
        1: class_weight_1
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
    losses['batch_size_test'] = test_batch_size
    losses['augment_flip'] = augment_flip
    losses['augment_brightness'] = augment_brightness
    identifier = round(time())
    test_indices_flat = [i for batch in test_indices for i in batch]
    predictions_to_store = test_data.iloc[test_indices_flat].drop('image', axis=1).copy()
    predictions_to_store['prediction'] = model.predict_generator(
        test_generator,
        steps=num_batches_test
    )
    losses['recall_val'] = (predictions_to_store.loc[
        predictions_to_store['destroyed'] == 1,
        'prediction'
    ] > 0.5).mean()
    losses['precision_val'] = predictions_to_store.loc[
        predictions_to_store['prediction'] >= 0.5,
        'destroyed'
    ].mean()
    losses['roc_auc_val'] = roc_auc_score(
        predictions_to_store['destroyed'], 
        predictions_to_store['prediction'],
    )
    with open('{}/experiment_{}.json'.format(EXPERIMENTS_PATH, identifier), 'w') as f:
        json.dump(str(losses), f)
        print('Experiment saved in experiment_{}.json'.format(identifier))
        
    if losses['roc_auc_val'] > 0.8:
        predictions_to_store.to_pickle(
            '{}/test_set_{}.p'.format(PREDICTIONS_PATH, identifier)
        )
        model.save('{}/model_{}.h5'.format(MODELS_PATH, identifier))
