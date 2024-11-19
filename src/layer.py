from activations import *

class Layer:
    def __init__(self, num_units, activation_fun):
        self._num_units = num_units
        self._activation_fun = activation_fun

    def get_num_units(self):
        return self._num_units

    def get_activation_fun(self):
        return self._activation_fun

    def activate(self, Z):
        if self._activation_fun == 'relu':
            return relu(Z)
        if self._activation_fun == 'sigmoid':
            return sigmoid(Z)
        if self._activation_fun == 'softmax':
            return softmax(Z)
        
    def activate_deriv(self, Z):
        if self._activation_fun == 'relu':
            return deriv_relu(Z)
        if self._activation_fun == 'sigmoid':
            return deriv_sigmoid(Z)


    