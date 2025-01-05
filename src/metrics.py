import numpy as np


def accuracy_bin_classification(A, Y):
    """
    Calculate the accuracy for binary classification.

    Parameters:
    A (numpy.ndarray): The predicted probabilities for the positive class.
    Y (numpy.ndarray): The true labels (0 or 1).

    Returns:
    float: The accuracy of the predictions as a percentage.
    """
    predicted_classes = (A >= 0.5).astype(int)
    return np.mean(predicted_classes == Y.T) * 100

def mse(A, Y):
    """
    Calculate the Mean Squared Error (MSE) between the predicted values and the actual values.

    Parameters:
    A (numpy.ndarray): Predicted values.
    Y (numpy.ndarray): Actual values.

    Returns:
    float: The mean squared error between the predicted and actual values.
    """
    return np.mean(np.square(A - Y.T))

def deriv_mse(A, Y):
    """
    Compute the derivative of the Mean Squared Error (MSE) loss function.

    Parameters:
    A (numpy.ndarray): The predicted values.
    Y (numpy.ndarray): The true values.

    Returns:
    numpy.ndarray: The derivative of the MSE loss function with respect to the predictions.
    """
    m = Y.size
    return 2/m * (A - Y.T)

def mee(A, Y):
    """
    Mean Euclidean Error (MEE) metric.

    This function calculates the Mean Euclidean Error between the predicted values (A) 
    and the actual values (Y).

    Parameters:
    A (numpy.ndarray): Predicted values, shape (n_samples, n_features).
    Y (numpy.ndarray): Actual values, shape (n_features, n_samples).

    Returns:
    float: The mean Euclidean error.
    """
    return np.mean(np.sqrt(np.sum((A - Y.T) ** 2, axis=1)))

def metric_acquisition(A, Y, metric):
    """
    Computes the specified metric between the predicted values (A) and the true values (Y).

    Parameters:
    A (array-like): Predicted values.
    Y (array-like): True values.
    metric (str): The metric to compute. Supported metrics are:
                  - 'acc_bin': Binary classification accuracy.
                  - 'mse': Mean Squared Error.
                  - 'mee': Mean Euclidean Error.

    Returns:
    float: The computed metric value.

    Raises:
    ValueError: If an invalid metric is specified.
    """
    if metric == 'acc_bin':
        return accuracy_bin_classification(A, Y)
    elif metric == 'mse':
        return mse(A, Y)
    elif metric == 'mee':
        return mee(A, Y)
    else:
        raise ValueError('Invalid Metric')

def error_computation(A, Y, err_fun):
    """
    Computes the error between the predicted values and the actual values using the specified error function.

    Parameters:
    A (array-like): Predicted values.
    Y (array-like): Actual values.
    err_fun (str): The error function to use. Currently supports 'mse' (mean squared error).

    Returns:
    float: The computed error.

    Raises:
    ValueError: If an invalid error function is specified.
    """
    if err_fun == 'mse':
        return deriv_mse(A, Y)
    else:
        raise ValueError('Invalid Error Function')
    
