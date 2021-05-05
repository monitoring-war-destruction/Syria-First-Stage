import argparse
import sys
from time import time
from datetime import timedelta
sys.path.insert(1, './')

from damage.data import load_data_multiple_cities, save_as_pickle_to_path
from damage import features
from damage.constants import FEATURES_PATH


# Create a parser to input data from the command line

parser = argparse.ArgumentParser(
    description='Process data for a given city and store it under {}/[filename]'\
        .format(FEATURES_PATH)
)
parser.add_argument(
    '--filename',
     help='''Features and target will be stored under {}/[filename].
             It needs to end with ".p" as it will be stored as a pickle file
             (example: aleppo.p)''',
)
parser.add_argument(
    '--city',
    help='city name to process the data for (e.g.: aleppo)',
    required=True,
)
args = vars(parser.parse_args())
file_name = args.get('filename', None) or '{}.p'.format(str(round(time())))
assert file_name.endswith('.p'), 'ERROR: filename needs to end with ".p"'

# Constants
CITIES = [args['city']]
PATCH_SIZE = 64 # size of the patch (image)
TIME_TO_ANNOTATION_THRESHOLD = timedelta(weeks=1)
STRIDE = PATCH_SIZE  # dont change

# Reading
print('Reading data for cities {}'.format(CITIES))
data = load_data_multiple_cities(CITIES)

# Processing
print('Features will be computed for {}'.format(CITIES))
pipeline = features.Pipeline(
    preprocessors=[
        ('AnnotationPreprocessor', features.AnnotationPreprocessor()),
    ],
    features=[
        ('RasterSplitter', features.RasterSplitter(patch_size=PATCH_SIZE, stride=STRIDE)),
        ('AnnotationMaker', features.AnnotationMaker_fillzeros(PATCH_SIZE, TIME_TO_ANNOTATION_THRESHOLD)),
        ('RasterPairMaker', features.RasterPairMaker()),
    ],
)
features = pipeline.transform(data)

# Logging results
print('Resulting features had {} rows and {} columns'.format(features.shape[0], features.shape[1]))
print('Column names are: {}'.format(features.columns))
print('Missing values per column:')
print(features.isnull().sum())
print('Proportion of missing values per column:')
print(features.isnull().mean())
print('This is how the first 5 rows look like:')
print(features.head())

# Storing
print('Features will now be stored')
save_as_pickle_to_path(features, '{}/{}'.format(FEATURES_PATH, file_name))
print('Features stored in {}/{}'.format(FEATURES_PATH, file_name))
print('Target will now be stored')
save_as_pickle_to_path(features[['destroyed', 'latitude', 'longitude', 'no_analysis']],
                       '{}/target_{}'.format(FEATURES_PATH, file_name))
print('Target stored in {}/target_{}'.format(FEATURES_PATH, file_name))
