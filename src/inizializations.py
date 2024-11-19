import numpy as np

def uniform_init(n, m):
    return np.random.uniform(-0.7,0.7,(n, m))

def init_weights(mode, n , m):
    if mode == 'uniform':
        return uniform_init(n, m)
    
