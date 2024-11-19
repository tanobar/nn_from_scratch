import numpy as np

def one_hot(Y):
    one_hot = np.zeros((Y.size, Y.max() + 1))
    one_hot[np.arange(Y.size), Y.astype(int)] = 1
    return one_hot.T
