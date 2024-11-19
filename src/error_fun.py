import numpy as np
from utils import one_hot

def mse(A, Y):
    return np.mean(np.square(A - Y.T))

def mse_deriv(A, Y):
    m = Y.size
    return (A - Y.T) / m

def cross_entropy_loss(A, Y):
    """
    Calcola la Cross-Entropy Loss.
    
    Parametri:
        A: numpy array, array delle probabilità previste (shape: [N, C]).
        Y: numpy array, array delle etichette one-hot (shape: [N, C]).

    Ritorna:
        float, il valore medio della Cross-Entropy Loss.
    """
    epsilon = 1e-15  # avoid log(0)
    A = np.clip(A, epsilon, 1 - epsilon)  # limits values of A
    return -np.mean(np.sum(Y * np.log(A), axis=1))

def error_comp(A, Y, task):
    if task == 'classification':
        return cross_entropy_loss(A, one_hot(Y))

    if task == 'regression':
        return mse_deriv(A, Y)

