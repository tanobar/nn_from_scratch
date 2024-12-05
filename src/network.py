import yaml
import numpy as np
from layer import Layer
from inizializations import *

class InvalidHyperparameterError(Exception):
    # custom exception for invalid hyperparameter values
    pass

class Net:
    def __init__(self, config_path):
        self._hyperparameters = self._load_config(config_path)
        self._validate_hyperparameters()
        self._layers = []
        self._W = []
        self._b = []

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
            "task": ["bin_classification", "multi_classification", "regression"],
            "initializer": ["uniform", "xavier", "he"],
            "momentum": [True, False],
            "optimizer": ["gd"]
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
        
        if "alpha" in self._hyperparameters and not (0.5 <= self._hyperparameters["alpha"] <= 0.9):
            raise InvalidHyperparameterError(f"'alpha' must be between 0.5 and 0.9. Got {type(self._hyperparameters['alpha'])}.")


    def get_hyperparameters(self):
        return self._hyperparameters

    def print_hyperparameters(self):
        # print the hyperparameters as a dictionary
        print(self._hyperparameters)

    def get_num_features(self):
        return self._hyperparameters['input_features']

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
            self._W.append(init_weights(self._hyperparameters['initializer'], num_units, self.get_num_features()))
            self._b.append(np.zeros((num_units, 1)))
        else:
            self._W.append(init_weights(self._hyperparameters['initializer'], num_units, self.get_num_inputs()))
            self._b.append(np.zeros((num_units, 1)))

