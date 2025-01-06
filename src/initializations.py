import numpy as np

def uniform_init(n, m):
    """
    Initialize weights uniformly between -0.7 and 0.7.

    Parameters:
    - n: Number of rows.
    - m: Number of columns.

    Returns:
    - Initialized weight matrix.
    """
    return np.random.uniform(-0.7, 0.7, (n, m))

def xavier_init(n, m):
    """
    Initialize weights using Xavier initialization.

    Parameters:
    - n: Number of rows.
    - m: Number of columns.

    Returns:
    - Initialized weight matrix.
    """
    limit = np.sqrt(6 / (n + m))
    return np.random.uniform(-limit, limit, (n, m))

def he_init(n, m):
    """
    Initialize weights using He initialization.

    Parameters:
    - n: Number of rows.
    - m: Number of columns.

    Returns:
    - Initialized weight matrix.
    """
    limit = np.sqrt(2 / n)
    return np.random.uniform(-limit, limit, (n, m))

def init_weights(mode, n, m):
    """
    Initialize weights based on the specified mode.

    Parameters:
    - mode: Initialization mode ('uniform' or 'xavier' or 'he').
    - n: Number of rows.
    - m: Number of columns.

    Returns:
    - Initialized weight matrix.
    """
    if mode == 'uniform':
        return uniform_init(n, m)
    elif mode == 'xavier':
        return xavier_init(n, m)
    elif mode == 'he':
        return he_init(n, m)
    else:
        raise ValueError("Unsupported initialization mode: {}".format(mode))
