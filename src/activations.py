import numpy as np

def relu(Z):
    """
    Applies the Rectified Linear Unit (ReLU) activation function to the input.

    The ReLU function is defined as:
        ReLU(x) = max(0, x)
    It sets all negative values in the input to zero and keeps all positive values unchanged.

    Parameters:
    Z (numpy.ndarray): Input array (can be a vector or matrix) to which the ReLU function is applied.

    Returns:
    numpy.ndarray: Output array with the ReLU function applied element-wise.
    """
    return np.maximum(0, Z)

def deriv_relu(Z):
    """
    Compute the derivative of the ReLU activation function.

    Parameters:
    Z (numpy.ndarray): Input array for which to compute the derivative.

    Returns:
    numpy.ndarray: An array where each element is 1 if the corresponding element in Z is greater than 0, otherwise 0.
    """
    return Z > 0

def sigmoid(Z):
    """
    Compute the sigmoid of Z.

    Parameters:
    Z (numpy.ndarray): Input array or matrix.

    Returns:
    numpy.ndarray: The sigmoid of the input, element-wise.
    """
    return 1 / (1 + np.exp(-Z.astype(float)))

def deriv_sigmoid(Z):
    """
    Compute the derivative of the sigmoid function.

    The derivative of the sigmoid function, given by the formula:
    sigmoid'(Z) = sigmoid(Z) * (1 - sigmoid(Z))
    where sigmoid(Z) = 1 / (1 + exp(-Z)).

    Parameters:
    Z (numpy.ndarray): The input array for which to compute the derivative of the sigmoid function.

    Returns:
    numpy.ndarray: The derivative of the sigmoid function applied element-wise to the input array Z.
    """
    return Z * (1 - Z)

def identity(Z):
    """
    Identity activation function.

    This function returns the input as is, without any transformation.

    Parameters:
    Z (numpy.ndarray): The input array.

    Returns:
    numpy.ndarray: The same input array.
    """
    return Z

def deriv_identity(Z):
    """
    Compute the derivative of the identity activation function.

    The identity activation function is f(x) = x, and its derivative is always 1.
    This function returns an array of ones with the same shape as the input array Z.

    Parameters:
    Z (numpy.ndarray): Input array for which the derivative is to be computed.

    Returns:
    numpy.ndarray: An array of ones with the same shape as Z.
    """
    return np.ones_like(Z)

