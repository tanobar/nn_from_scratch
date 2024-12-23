import numpy as np

def relu(Z):
    return np.maximum(0, Z) # it is element wise, so we can use vectors or matrixes as input

def deriv_relu(Z):
    return Z > 0

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z.astype(float)))

def deriv_sigmoid(Z):
    return Z * (1 - Z)

def identity(Z):
    return Z

def deriv_identity(Z):
    return np.ones_like(Z)

