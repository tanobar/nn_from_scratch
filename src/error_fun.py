import numpy as np
from utils import one_hot_vector

def mse(A, Y):
    return np.mean(np.square(A - Y.T))

def deriv_mse(A, Y):
    m = Y.size
    return 2/m * (A - Y.T)

def error_computation(A, Y, task):
    if task == 'bin_classification':
        return deriv_mse(A, Y)

    if task == 'regression':
        return deriv_mse(A, Y)

