import random
import numpy as np


class RandomSearch:

    def sample_cnn(self, T):
        spaces = [self._sample_single_cnn_space() for _ in range(T)]
        return spaces

    def sample_cnn_pretrained(self, T):
        spaces = [self._sample_single_cnn_pretrained_space() for _ in range(T)]
        return spaces
    
    @staticmethod 
    def _sample_convolutional_layers(num_layers):
        convolutional_layers = []
        filters = random.choice([8, 16, 32, 64])
        kernel_size = random.choice([3, 5, 7, 9])
        pool_size = kernel_size-1
        dropout = random.choice(np.linspace(0.1, 0.8, 10))
        activation = random.choice(['relu']) 
        for _ in range(num_layers):
            filters = filters*2
            layer = {
                'kernel_size': [kernel_size, kernel_size],
                'pool_size': [pool_size, pool_size],
                'filters': filters,
                'dropout': dropout,
                'activation': activation, 
            }
            convolutional_layers.append(layer)

        return convolutional_layers

    @staticmethod
    def _sample_single_cnn_space():
        num_layers = random.choice(range(2, 6))
        convolutional_layers = RandomSearch._sample_convolutional_layers(num_layers)
        total_filters = sum(layer['filters'] for layer in convolutional_layers)
        while total_filters > 1800:
            convolutional_layers = RandomSearch._sample_convolutional_layers(num_layers)
            total_filters = sum(layer['filters'] for layer in convolutional_layers)

        space = {
            'dense_units': random.choice([16, 32, 64, 128, 256]),
            'batch_size': random.choice(range(150, 225)),
            'convolutional_layers': convolutional_layers,
            'epochs': random.choice(range(5, 15)),
            'layer_type': random.choice(['cnn']),
            'class_weight': random.choice(np.linspace(0.2, 1.7)),
            'learning_rate': random.choice(np.linspace(0.6, 1)),
            'activation_output': random.choice(['relu']) # ReLU has shown much better performance
        }

        return space

    @staticmethod
    def _sample_single_cnn_pretrained_space():
        num_layers = random.choice(range(3))
        convolutional_layers = RandomSearch._sample_convolutional_layers(num_layers)
        space = {
            'dense_units': random.choice([32, 64, 128, 256, 512, 1024]),
            'learning_rate': random.choice(np.linspace(0.6, 1)),
            'batch_size': random.choice(range(8, 25)),
            'epochs': random.choice(range(5, 15)),
            'class_weight': random.choice(np.linspace(0.2, 1.7)),
            'convolutional_layers': convolutional_layers,
        }
        return space
