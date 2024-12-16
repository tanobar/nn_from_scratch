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
            "err_fun": ["mse"],
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

        # Check for layers_config
        if "layers_config" in self._hyperparameters:
            layers_config = self._hyperparameters["layers_config"]
            if "n_layers" not in layers_config or not isinstance(layers_config["n_layers"], int) or layers_config["n_layers"] <= 0:
                raise InvalidHyperparameterError(f"'n_layers' must be a positive integer. Got {layers_config.get('n_layers')}.")

            if "units" not in layers_config or not isinstance(layers_config["units"], list) or len(layers_config["units"]) != layers_config["n_layers"]:
                raise InvalidHyperparameterError(f"'units' must be a list of length 'n_layers'. Got {layers_config.get('units')} with length {len(layers_config.get('units', []))}.")            
            # Check that every entry in the units list is greater than 0
            if any(unit <= 0 for unit in layers_config["units"]):
                raise InvalidHyperparameterError(f"All entries in 'units' must be greater than 0. Got {layers_config['units']}.")

            if "activations" not in layers_config or not isinstance(layers_config["activations"], list) or len(layers_config["activations"]) != layers_config["n_layers"]:
                raise InvalidHyperparameterError(f"'activations' must be a list of length 'n_layers'. Got {layers_config.get('activations')} with length {len(layers_config.get('activations', []))}.")

            for activation in layers_config["activations"]:
                if activation not in ["relu", "sigmoid"]: # Add more activations if needed
                    raise InvalidHyperparameterError(f"Invalid activation '{activation}'. Allowed values are: {allowed_values['activations']}.")


    def add_layer(self, num_units, activation): # add manually a layer to the network
        self._layers.append(Layer(num_units, activation))
        if len(self._W) == 0: # it means that is the first hidden layer and recives as input the feature matrix X
            self._W.append(init_weights(self._hyperparameters['initializer'], num_units, self.get_num_features()))
            self._b.append(np.zeros((num_units, 1)))
        else:
            self._W.append(init_weights(self._hyperparameters['initializer'], num_units, self.get_num_inputs()))
            self._b.append(np.zeros((num_units, 1)))


    def build_net(self): # build the network based on the layers_config
        if 'layers_config' in self._hyperparameters:
            layers_config = self._hyperparameters['layers_config']
            for k in range(layers_config['n_layers']):
                units = layers_config['units'][k]
                activation = layers_config['activations'][k]
                self.add_layer(units, activation)


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
        # -2 access the second-last layer, it depends on operations order
        return self.get_layers()[len(self._layers)-2].get_num_units()

    def print_structure(self):
        for i, layer in enumerate(self._layers):
            print(f"Layer {i + 1}: Units = {layer.get_num_units()}, Activation = {layer.get_activation_fun()}")

