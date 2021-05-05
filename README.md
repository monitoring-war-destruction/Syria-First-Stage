# Syria-First-Stage
Repository containing code for infering building damage from satellite imaging

## Installation

To get started write the following commands in your terminal:

```
git clone https://github.com/javiermas/building-damage.git
cd building-damage
pip install -r requirements.txt
pip install -e .
```
...and you're good to go!

## Usage

The pipeline is structured in three steps represented by three scripts: `compute_features.py`, `validate.py`, `generate_dense_prediction_single_city.py`.

### Computing features

To compute features you need to run the following command:
```
python scripts/compute_features.py --filename=<example_name.p>
```
This command will store two pickle objects: `example_name.p` and `target_example_name.p` under `logs/features`. The first file __ALSO__ contains the target, but given that it will a very large file, we also store the target individually so it can be easily sent from a remote machine (server) to a local machine (your own laptop) for exploration purposes. The folder where we store these files can be modified on `scripts/compute_features.py`.

### Running experiments

To run experiments and explore the performance of the models with the created features you need to run the following command:
```
python scripts/validate.py --features=<example_name.p> --gpu=<gpu_number>
```
By default, this script will run 500 experiments, storing the hyperparameter space and loss after every iteration and the entire model if the precision and true positive rate on validation are above 0.1 and 0.4 on the last epoch. You can stop it at any time. Note that the script will look for the given file under `logs/features`. The experiments will stored in `logs/experiments` using the name `experiment_<timestamp>.json`. The models will be stored in `logs/models` using the name `model_<timestamp>.h5`, that way the model can be matched to the corresponding experiment.

In case you want to run multiple experiments in parallel with different GPUs (or the same if data is small), you need to run the following command:
```
bash scripts/parallel_validate.sh <example_name.p> <gpu_0> ... <gpu_N>
```
This command will run N instances of `scripts/validate.py` in parallel using the given file name and the specificed GPUs.

### Generating predictions

To generate predictions for all data you need to run the following command:
```
python scripts/generate_prediction_single_city.py --features=<example_name.p> --gpu=<gpu_number>
```
This script will train on all data and predict on all data to generate maps of destruction for any given city and date. If you want to split the data, train on one part and predict on another one, you need to run the following command:
```
python scripts/predict.py --features=<example_name.p> --gpu=<gpu_number>
```
Predictions will be stored as pickle files under `logs/predictions` using the name `prediction_<timestamp>.p`

### Tutorials

For more detailed information on the process, you can check the tutorial notebooks under notebooks. Bear in mind they might not be 100\% up to date.

## Technical details

The architecture is designed on three modules: data reading, data processing and data modelling. Following these modules, there are three folders inside of the python library. The folders _data_, _features_ and _modelling_ contain functions and classes with the responsibility of reading, processing and modelling data, respectively.

### Data reading
The reading functions assume a certain structure in the data to ease the interaction with them. Every city is supposed to have three types of files: annotations (.shp), rasters (.tif), no-analysis areas (.shp) and populated areas (.shp). The default path to these are `data/annotations`, `data/city_rasters` and `data/polygons` but you can provide different ones to the reading functions. The file names corresponding to each city must be stored in the file `damage/data/data_sources.py`, which contains a python dictionary following the form: 
```
{
    <city> : {
        annotations: [
            annotation_file.shp
       ]},
       rasters: [
            raster_date_0.tif,
            ..., 
            raster_date_N.tif
       ],
       no_analysis: [
            no_analysis_file.shp
       ]
}
```
This way, one only needs to pass a list of cities to `load_data_multiple_cities` to retrieve all the data in a dictionary with the file name as a key and the data as a value.

### Data processing
Data is processed using functions or _Transformer_ classes that get passed to the _Pipeline_ object. A Transformer class is a class that has a _transform_ method with a data argument. That argument receives a data object, which is a dictionary of this form:
```
{
    key_0: pandas.DataFrame(),
    ...
    key_N: pandas.DataFrame()
}
```
_Preprocessor_ classes receive that data dictionary and return it with some modifications. Note that _Pipeline_ overwrites the data object when iterating over preprocessors.
_Feature_ classes receive that dictionary and return a pandas.DataFrame() indexed by 'city', 'patch_id' and 'date'.Note that _Pipeline_ adds a new key to the data dictionary with the feature name that was passed. Once the _Pipeline_ object has finished iterating over preprocessors and features, it merges the dataframes that came out of the feature classes, making use of their indices. Remember that features need to return a pandas.DataFrame that is indexed by 'city', 'patch_id' and 'date'. This structure makes further manipulation of the data very easy.

The current pipeline is composed of 4 transformers:
\n
1. __AnnotationPreprocessor__: This preprocessor transforms annotations, adding latitude and longitudes, turning annotations to numerical values, cropping them to be within image coordinates and reshaping the format for later use.
2. __RasterSplitter__: This feature class transformrs the rasters, splitting them into square patches of a given size. These will be the base images that we will try to classify.
3. __AnnotationMaker__: This feature class matches the annotations with the raster patches coordinates, rolling forward the destruction and back the non-destruction (assumption: destruction is incremental).
4. __RasterPairMaker__: This feature class takes the raster patches and concatenates them with their matching pre-war patch over the third axis. That way we will end up with patches/tensors of size patch_size x patch_size x 6. We expect the NN to learn to distinguish destruction by looking at the difference between a pre-war and post-war image for a given part of the city. These are the final images to be classified. 

The resulting dataframe will look like this:

| image       | patch_id | date   | latitude    | longitude   | destroyed |
| ----------- | -------- | ------ | ----------- | ----------- | --------- |
| image_0     | 0-10     | date_0 | latitude_0  | longitude_0 | 0         |
| ...         | ...      | ...    | ...         | ...         | ...       |
| image_N     | w-h      | date_N | latitude_N  | longitude_N | 1         |


### Modelling

Models used can be found under `damage/models`. In practice, we use a CNN class that can be customized through the arguments passed to the constructor function. In order to explore the hyperparameter space we created a RandomSearch object that samples random hyperparameter spaces, following some predetermined distributions (for see `damage/models/random_search.py`). With respect to the convolutional layers, we sample the number of units in the first convolutional layer. That number then gets mutiplied by a factor of two for every posterior layer. The class weight hyperparameter multiplies a default class weight of 0.7 (1s) and 0.3 (0s) that corresponds approximately to the proportion of 1s and 0s in the training data after upsampling the 1s.


__The coordinates problem__

Some .tiff files seem to have either the latitude or longitude constant across the entire image. This renders them useless because we can't link them to the annotation, where the damage levels are associated to a specific coordinate. See the following notebook for the corresponding analysis:

https://github.com/javiermas/building-damage/blob/master/notebooks/jma_02_coordinate_problems.ipynb

It turns out that they can be reconstructed by using the corresponding .bmp files.
