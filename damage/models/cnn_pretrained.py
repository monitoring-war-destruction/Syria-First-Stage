from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19

from damage.models.losses import *
from damage.models.base import CNNModel


class CNNPreTrained(CNNModel):

    metrics = ['accuracy', recall_positives, recall_negatives, precision_positives,
               precision_negatives, negatives, positives, true_positives,
               true_negatives, false_positives, false_negatives]

    def __init__(self, pre_trained_model=None, convolutional_layers=[], 
                 dense_units=64, learning_rate=0.1, **kwargs):
        self.pre_trained_model = pre_trained_model or VGG19(include_top=False, weights='imagenet')
        for layer in self.pre_trained_model.layers:
            layer.trainable = False
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.convolutional_layers = convolutional_layers
        self.model = self._create_model()

    def _create_model(self):
        input_layer = Input(shape=(64, 64, 6))
        conv_layer = Conv2D(3, 3, padding='same')(input_layer)
        x = self.pre_trained_model(conv_layer)

        for config in self.convolutional_layers:
            _layers = self._create_convolutional_and_pooling_layer_cnn(**config)
            for _layer in _layers:
                x = _layer(x)

        x = GlobalAveragePooling2D()(x)
        x = Dense(units=self.dense_units, activation='relu')(x)
        predictions = Dense(units=1, activation='relu')(x)
        model = Model(inputs=input_layer, outputs=predictions)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=self.metrics)
        return model
