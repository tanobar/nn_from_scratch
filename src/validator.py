import numpy as np
import yaml
from tqdm import tqdm
from sklearn.model_selection import KFold, ParameterGrid
from trainers import *
from joblib import Parallel, delayed

class Validator:
    def __init__(self, grid):

        with open(grid, 'r') as file:
            self.param_grid = yaml.safe_load(file)

        self.grid = list(ParameterGrid(self.param_grid))
        # remove from self.grid all the configurations with len(units) != len(activations)
        self.grid = [conf for conf in self.grid if len(conf['units']) == len(conf['activations'])]

        self.kfold = KFold(n_splits=3, shuffle=True, random_state=42)
        self.early_stopping_patience = 10  # Number of epochs to wait for improvement before stopping


    def train_model_ES(self, X_train, Y_train, X_val, Y_val, network):
        best_weights = None
        best_biases = None
        best_epoch = None
        best_val_loss = float('inf')
        patience_counter = 0

        candidate = [network.get_W(), network.get_b(), 1]
        network.get_hyperparameters()['epochs'] = 1

        for epoch in range(1000000):
            candidate = train_model(X_train, Y_train, candidate[0], candidate[1], network.get_layers(), network.get_hyperparameters())
            val_loss = test_model(X_val, Y_val, candidate[0], candidate[1], network.get_layers(), 'mse')

            if val_loss < best_val_loss * 0.99:  # 1% improvement criteria
                best_val_loss = val_loss
                best_weights = candidate[0]
                best_biases = candidate[1]
                best_epoch = epoch  # Update best_epoch when improvement is found
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                break

        return best_weights, best_biases, best_epoch
    

    def grid_search(self, X, Y, network):
        def evaluate_configuration(conf):
            conf_metrics = []
            for train_index, val_index in self.kfold.split(X):
                X_train, X_val = X[train_index], X[val_index]
                Y_train, Y_val = Y[train_index], Y[val_index]

                network.rebuild_net(conf)
                candidate = self.train_model_ES(X_train, Y_train, X_val, Y_val, network)
                m = test_model(X_val, Y_val, candidate[0], candidate[1], network.get_layers(), network.get_hyperparameters()['metric'])
                conf_metrics.append(m)
            if not np.isnan(conf_metrics).any():
                return {
                    'conf': conf,
                    'candidate': candidate,
                    'metrics': np.mean(conf_metrics)
                }
            return None

        search_metrics = Parallel(n_jobs=-1)(delayed(evaluate_configuration)(conf) for conf in tqdm(self.grid))
        search_metrics = [metric for metric in search_metrics if metric is not None]

        if network.get_hyperparameters()['metric'] == 'acc_bin':
            best_conf = max(search_metrics, key=lambda x: x['metrics'])
        else:
            best_conf = min(search_metrics, key=lambda x: x['metrics'])
        return best_conf


