import numpy as np

def mse(A, Y):
    return np.mean(np.square(A - Y.T))

def deriv_mse(A, Y):
    m = Y.size
    return 2/m * (A - Y.T)

def error_computation(A, Y, err_fun):
    if err_fun == 'mse':
        return deriv_mse(A, Y)
    else:
        raise ValueError('Invalid Error Function')

