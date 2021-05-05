from math import ceil
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


class EnsembleSplitter:
    def __init__(self, batch_size, test_batch_size, train_proportion, class_proportion=None):
        self.batch_size = batch_size
        self.test_batch_size= test_batch_size
        self.train_proportion = train_proportion
        self.class_proportion = class_proportion

    def split_by_patch_id(self, data):
        patch_mappings = self._get_patch_mappings(data)
        largest_patch_size = max(list(data.keys()))
        unique_patches = data[largest_patch_size].index.get_level_values('patch_id').unique().tolist()
        train_patches = random.sample(unique_patches, round(len(unique_patches)*self.train_proportion))
        test_patches = list(set(unique_patches) - set(train_patches))
        batches = {}
        import ipdb; ipdb.set_trace()
        for key, value in data.items():
            value['patch_id_largest'] = value.index.get_level_values('patch_id')\
                .map(lambda x: patch_mappings[x])
            train_data = value.loc[value['patch_id_largest'].isin(train_patches)]
            if self.class_proportion is not None:
                train_data = self._upsample_class_proportion(train_data).sample(frac=1)

            train_index_generator = self._get_index_generator(train_data, self.batch_size)
            test_data = value.loc[value['patch_id_largest'].isin(test_patches)]
            test_index_generator = self._get_index_generator(test_data, self.test_batch_size)
            batches[key] = (train_index_generator, test_index_generator)

        return batches

    def _get_index_generator(self, features, batch_size):
        num_batches = ceil(len(features) / batch_size)
        stratified_k_fold = StratifiedKFold(n_splits=num_batches)
        batches = stratified_k_fold.split(features['destroyed'], features['destroyed'])
        batches = [features.iloc[batch[1]].index for batch in batches]
        return batches

    @staticmethod
    def get_train_data_generator_from_index(data, index):
        while True:
            for _index in index:
                batch = tuple([np.stack(dataframe.loc[_index]) for dataframe in data])
                yield batch

    @staticmethod
    def get_test_data_generator_from_index(data, index):
        while True:
            for _index in index:
                batch = np.stack(data.loc[_index]) 
                yield batch

    def _upsample_class_proportion(self, data):
        current_proportion = data['destroyed'].mean()
        assert self.class_proportion[1] > current_proportion
        data_non_destroyed = data.loc[data['destroyed'] == 0]
        data_destroyed = data.loc[data['destroyed'] == 1]
        data_destroyed_upsampled = data_destroyed.sample(
            int(len(data_non_destroyed)*self.class_proportion[1]),
            replace=True
        )
        data_resampled = pd.concat([data_non_destroyed, data_destroyed_upsampled])
        return data_resampled

    def _get_patch_mappings(self, data):
        largest_patch_size = max(list(data.keys()))
        patch_mappings = {}
        import ipdb; ipdb.set_trace()
        for patch_size, level in data.items():
            indices = level.index.get_level_values('patch_id')
            cols_rows = [elem.split('-') for elem in indices]
            cols, rows = list(map(list, zip(*cols_rows)))
            rows_centroids = [int(largest_patch_size * (int(r)//largest_patch_size) + (largest_patch_size/2))
                              for r in rows]
            cols_centroids = [int(largest_patch_size * (int(c)//largest_patch_size) + (largest_patch_size/2))
                              for c in cols]
            patch_mapping = {index: '{}-{}'.format(col, row) for index, row, col
                             in zip(indices, rows_centroids, cols_centroids)}
            patch_mappings = {**patch_mappings, **patch_mapping}

        return patch_mappings 

