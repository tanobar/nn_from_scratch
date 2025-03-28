import yaml
import numpy as np
from layer import Layer
from initializations import *


class InvalidHyperparameterError(Exception):
    # custom exception for invalid hyperparameter values
    pass


class Net:
    """
    A class used to represent a Neural Network.
    Attributes
    ----------
    _hyperparameters : dict
        A dictionary containing the hyperparameters of the network.
    _layers : list
        A list containing the layers of the network.
    _W : list
        A list containing the weight matrices of the network.
    _b : list
        A list containing the bias vectors of the network.
    Methods
    -------
    __init__(config_path)
        Initializes the network with the given configuration file.
    _load_config(config_path)
        Loads hyperparameters from a YAML configuration file.
    _validate_hyperparameters()
        Validates the hyperparameter values.
    add_layer(num_units, activation)
        Adds a layer to the network manually.
    build_net()
        Builds the network based on units and activations in the hyperparameters.
    rebuild_net(hyperparameters)
        Rebuilds the network with new hyperparameters.
    set_best_configuration(best)
        Sets the best configuration for the network.
    get_hyperparameters()
        Returns the hyperparameters of the network.
    get_num_features()
        Returns the number of input features.
    get_layers()
        Returns the layers of the network.
    get_W()
        Returns the weight matrices of the network.
    get_b()
        Returns the bias vectors of the network.
    get_num_inputs()
        Returns the number of inputs coming from the previous layer.
    print_structure()
        Prints the structure of the network.
    print_hyperparameters()
        Prints the hyperparameters of the network.
    """
    def __init__(self, config_path):
        self._hyperparameters = self._load_config(config_path)
        self._validate_hyperparameters()
        self._layers = []
        self._W = []
        self._b = []
        self.build_net()


    def _load_config(self, config_path):
        # load hyperparameters from a YAML configuration file
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The configuration file '{config_path}' was not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing the YAML configuration file: {e}")
        
        return config


    def _validate_hyperparameters(self):
        # validate the hyperparameter values. Modify allowed_values for adding/removing hyperparameters and allowed values
        allowed_values = {
            "err_fun": ["mse"],
            "metric": ["mee", "mse", "acc_bin"],
            "initializer": ["uniform", "xavier", "he"]
        }
        
        # Check each key-value pair
        for key, allowed in allowed_values.items():
            if key in self._hyperparameters:
                if self._hyperparameters[key] not in allowed:
                    raise InvalidHyperparameterError(
                        f"Invalid value '{self._hyperparameters[key]}' for '{key}'. "
                        f"Allowed values are: {allowed}."
                    )
            else:
                raise InvalidHyperparameterError(f"Missing required hyperparameter '{key}'.")

        # Additional checks for numeric or optional fields
        if "eta" in self._hyperparameters and not (0.0 < self._hyperparameters["eta"] < 1.0):
            raise InvalidHyperparameterError(f"'eta' must be between 0 and 1. Got {self._hyperparameters['eta']}.")
        
        if "epochs" in self._hyperparameters and not isinstance(self._hyperparameters["epochs"], int):
            raise InvalidHyperparameterError(f"'epochs' must be an integer. Got {type(self._hyperparameters['epochs'])}.")

        if "epochs" in self._hyperparameters and not (self._hyperparameters["epochs"] > 0):
            raise InvalidHyperparameterError(f"'epochs' must be greater than 0. Got {type(self._hyperparameters['epochs'])}.")

        if "units" in self._hyperparameters and len(self._hyperparameters["units"]) < 1:
            raise InvalidHyperparameterError(f"'units' must be a list of length greater than 0. Got {self._hyperparameters['units']} with length {len(self._hyperparameters['units'])}.")
                   
        # Check that every entry in the units list is greater than 0
        for unit in self._hyperparameters["units"]:
            if unit < 1:
                raise InvalidHyperparameterError(f"Invalid number of units '{unit}'. Must be greater than 0.")

        if "activations" in self._hyperparameters and len(self._hyperparameters["activations"]) != len(self._hyperparameters["units"]):
           raise InvalidHyperparameterError(f"'activations' must be a list of length equal to len(units). Got {self._hyperparameters['activations']} with length {len(self._hyperparameters['activations'])}.")
        
        for activation in self._hyperparameters["activations"]:
                if activation not in ["identity", "relu", "sigmoid"]: # Add more activations if needed
                    raise InvalidHyperparameterError(f"Invalid activation '{activation}'. Allowed values are: identity, relu, sigmoid.")


    def add_layer(self, num_units, activation):
        self._layers.append(Layer(num_units, activation))
        if len(self._W) == 0: # it means that is the first hidden layer and recives as input the feature matrix X
            self._W.append(init_weights(self._hyperparameters['initializer'], num_units, self.get_num_features()))
            self._b.append(np.zeros((num_units, 1)))
        else: # ops order: first append then init
            self._W.append(init_weights(self._hyperparameters['initializer'], num_units, self.get_num_inputs()))
            self._b.append(np.zeros((num_units, 1)))


    def build_net(self):
        if 'units' in self._hyperparameters and 'activations' in self._hyperparameters:
            units = self._hyperparameters['units']
            activations = self._hyperparameters['activations']
            for num_units, activation in zip(units, activations):
                self.add_layer(num_units, activation)


    def rebuild_net(self, hyperparameters):
        for key, value in hyperparameters.items():
            self._hyperparameters[key] = value
        self._hyperparameters['epochs'] = 1
        self._layers = []
        self._W = []
        self._b = []
        self.build_net()


    def set_best_configuration(self, best):
        self.rebuild_net(best['conf'])
        self._hyperparameters['epochs'] = best['epochs']


    def get_hyperparameters(self):
        return self._hyperparameters

    def get_num_features(self):
        return self._hyperparameters['input_features']

    def get_layers(self):
        return self._layers

    def get_W(self):
        return self._W

    def get_b(self):
        return self._b

    def get_num_inputs(self):
        # -2 access the second-last layer, it depends on python operations order: first append then init
        return self.get_layers()[len(self._layers)-2].get_num_units()

    def print_structure(self):
        for i, layer in enumerate(self._layers):
            print(f"Layer {i + 1}: Units = {layer.get_num_units()}, Activation = {layer.get_activation_fun()}")

    def print_hyperparameters(self):
        for key, value in self._hyperparameters.items():
            print(f"{key}: {value}")

