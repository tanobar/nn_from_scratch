import numpy as np
import matplotlib.pyplot as plt


def accuracy_bin_classification(A, Y):
    predicted_classes = (A >= 0.5).astype(int)
    return np.mean(predicted_classes == Y) * 100

def mse(A, Y):
    return np.mean(np.square(A - Y.T))

def deriv_mse(A, Y):
    m = Y.size
    return 2/m * (A - Y.T)

def mee(A, Y):
    return np.mean(np.sqrt(np.sum((A - Y.T) ** 2, axis=1)))

def accuracy(A, Y): # TODO change and modularize
    return accuracy_bin_classification(A, Y)

def error_computation(A, Y, err_fun):
    if err_fun == 'mse':
        return deriv_mse(A, Y)
    else:
        raise ValueError('Invalid Error Function')