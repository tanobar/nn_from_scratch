from activations import *

class Layer:
    """
    A class used to represent a Layer in a neural network.
    Attributes
    ----------
    _num_units : int
        The number of units (neurons) in the layer.
    _activation_fun : str
        The activation function used in the layer. Supported values are 'relu', 'sigmoid', and 'identity'.
    Methods
    -------
    get_num_units():
        Returns the number of units in the layer.
    get_activation_fun():
        Returns the activation function used in the layer.
    activate(Z):
        Applies the activation function to the input Z and returns the result.
    activate_deriv(Z):
        Applies the derivative of the activation function to the input Z and returns the result.
    """
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
        if self._activation_fun == 'identity':
            return Z
        raise ValueError(f"Unsupported activation function: {self._activation_fun}")
        
    def activate_deriv(self, Z):
        if self._activation_fun == 'relu':
            return deriv_relu(Z)
        if self._activation_fun == 'sigmoid':
            return deriv_sigmoid(Z)
        if self._activation_fun == 'identity':
            return deriv_identity(Z)
        raise ValueError(f"Unsupported activation function: {self._activation_fun}")

