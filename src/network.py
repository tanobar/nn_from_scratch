import numpy as np
from layer import Layer
from inizializations import *

class Net:
    def __init__(self): #TODO add params
        # hyperparameters
        self._task = 'bin_classification'
        self._input_features = 17
        self._inizializer = 'uniform'
        self._eta = 0.1
        self._momentum = 'none'
        self._epochs = 2500
        self._optimizer = "gd"


        self._layers = []
        self._W = []
        self._b = []

    def get_num_features(self):
        return self._input_features

    def get_layers(self):
        return self._layers

    def get_W(self):
        return self._W

    def get_b(self):
        return self._b

    def get_num_inputs(self):
        # -2 access the second-last layer, it depends on operations order
        return self.get_layers()[len(self._layers)-2].get_num_units()

    def print_structure(self):
        for i, layer in enumerate(self._layers):
            print(f"Layer {i + 1}: Units = {layer.get_num_units()}, Activation = {layer.get_activation_fun()}")

    def add_layer(self, num_units, activation):
        self._layers.append(Layer(num_units, activation))
        if len(self._W) == 0: # it means that is the first hidden layer and recives as input the feature matrix X
            self._W.append(init_weights(self._inizializer, num_units, self.get_num_features()))
            self._b.append(np.zeros((num_units, 1)))
        else:
            self._W.append(init_weights(self._inizializer, num_units, self.get_num_inputs()))
            self._b.append(np.zeros((num_units, 1)))

