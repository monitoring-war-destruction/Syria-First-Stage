from functools import reduce
import pandas as pd

from damage.features.base import Transformer


class Pipeline(Transformer):
    """ The Pipeline class takes a list of tuples (name, function)
    for preprocessors and features and applies them with
    the transform method. The feature functions need to return
    a dataframe with an identically structured index so the
    merge can be performed (e.g. city, patch_id)
    """

    def __init__(self, preprocessors, features):
        self.preprocessors = preprocessors
        self.features = features
        self.feature_names = [feature_name for feature_name, _ in self.features]

    def transform(self, data):
        """ The transform method takes a dictionary of data where
        each key represents a different data source (e.g. annotations)
        and each value a data object (e.g. pandas dataframe). The transform
        method iterates first over the preprocessor functions, overwriting
        the data object. Then, it iterates over the feature functions
        creating new keys in the data dictionary with the passed name.
        That way, features can use data generates by previously computed
        features.

        input
        -------
        data (dict): dictionary of data with data sources as keys
        and data objects as values.

        output
        -------
        feature_data (pandas.DataFrame): dataframe contained
        the merged data outputed by the feature functions.
        """

        for preprocessor_name, preprocessor in self.preprocessors:
            data = preprocessor(data)

        for feature_name, feature in self.features:
            data[feature_name] = feature(data)

        feature_data = [data[name] for name in self.feature_names if name in data.keys()]
        feature_data = self._merge_feature_data(feature_data)
        return feature_data

    def _merge_feature_data(self, feature_data):
        return reduce(lambda l, r: pd.merge(l, r, left_index=True, right_index=True, how='outer'), feature_data)
