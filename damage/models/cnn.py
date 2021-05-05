from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential

from damage.models.losses import *
from damage.models.base import CNNModel


class CNN(CNNModel):

    metrics = ['accuracy', recall_positives, recall_negatives, negatives, positives]

    def __init__(self, convolutional_layers, dense_units=64, learning_rate=0.1,
                 layer_type='cnn', activation_output='relu', **kwargs):
        self.convolutional_layers = convolutional_layers
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.layer_type = layer_type
        self.activation_output = activation_output
        self.layer_funcs = {
            'cnn': self._create_convolutional_and_pooling_layer_cnn,
            'vgg': self._create_convolutional_and_pooling_layer_vgg,
        }
        self._create_convolutional_and_pooling_layer = self.layer_funcs[self.layer_type]
        self.model = self._create_model()

    def _create_model(self):
        layers = []
        for config in self.convolutional_layers:
            layers.extend(self._create_convolutional_and_pooling_layer(**config))

        layers.extend([
            Flatten(),
            Dense(units=self.dense_units),
            BatchNormalization(),
            Dense(units=1, activation=self.activation_output),
        ])
        model = Sequential(layers)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      learning_rate=self.learning_rate,
                      metrics=self.metrics)
        return model
