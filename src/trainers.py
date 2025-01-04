import numpy as np
from metrics import *
from utils import *

def forward_prop(layers, W, b, X):
    Z, A = [None] * len(layers), [None] * len(layers)

    for i in range(len(layers)):
        Z[i] = W[i].dot(A[i-1] if i > 0 else X) + b[i]
        A[i] = layers[i].activate(Z[i])
    
    return Z, A


def back_prop(layers, err_fun, Z, A, W, X, Y):
    L = len(layers)
    dZ, dW, db = [None] * L, [None] * L, [None] * L
    
    # compute dZ for the last layer. By distributing 1/m before backprop (see error_computation()), the gradient with respect 
    # to weights and biases already includes the normalization factor, so (1/m) can be omitted during grandients and biases computation
    dZ[-1] = error_computation(A[-1], Y, err_fun)
    
    # loop backwards through layers to calculate gradients
    for i in reversed(range(L)):
        if i == L - 1:
            # output layer gradients
            dW[i] = dZ[i].dot(A[i-1].T)
            db[i] = np.sum(dZ[i], axis=1, keepdims=True)
        else:
            # hidden layer gradients
            dZ[i] = W[i + 1].T.dot(dZ[i+1]) * layers[i].activate_deriv(Z[i])
            dW[i] = dZ[i].dot(A[i-1].T if i > 0 else X.T)
            db[i] = np.sum(dZ[i], axis=1, keepdims=True)
    
    return dW, db


def update_params(num_layers, W, b, dW, db, eta, alpha, lambd, W_new, b_new):
    if alpha > 0:
        if W_new is None:
            W_new = [np.zeros_like(W[i]) for i in range(num_layers)]
        if b_new is None:
            b_new = [np.zeros_like(b[i]) for i in range(num_layers)]

        for i in range(num_layers):
            # momentum
            W_new[i] = -eta * dW[i] + alpha * W_new[i]
            b_new[i] = -eta * db[i] + alpha * b_new[i]

            if lambd > 0:
                W[i] = W[i] + W_new[i] - eta * lambd * W[i]
            else:
                W[i] = W[i] + W_new[i]
            b[i] = b[i] + b_new[i]

    else:
        for i in range(num_layers):
            # regularization
            if lambd > 0:
                W[i] = W[i] - eta * dW[i] - eta * lambd * W[i]
            else:
                W[i] = W[i] - eta * dW[i]
            b[i] = b[i] - eta * db[i]

    return W, b, W_new, b_new


# gradient descend algorithm
def grad_descent(X, Y, W, b, layers, hyperparameters):
    loss_data, metric_data = [], []
    W_new, b_new = None, None
    for i in range(hyperparameters['epochs']):
        Z, A = forward_prop(layers, W, b, X)
        dW, db = back_prop(layers, hyperparameters['err_fun'], Z, A, W, X, Y)
        W, b, W_new, b_new = update_params(len(layers), W, b, dW, db, hyperparameters['eta'],
                                            hyperparameters['alpha'], hyperparameters['lambd'], W_new, b_new)

        loss = mse(A[-1], Y)
        loss_data.append({'epoch': i, 'loss': loss})
        m = metric_acquisition(A[-1], Y, hyperparameters['metric'])
        metric_data.append(m)

    loss_save(loss_data)
    metric_save(metric_data, hyperparameters['metric'])

    return W, b


def train_model(X, Y, W, b, layers, hyperparameters):
    X = X.T
    model = []
    W, b = grad_descent(X, Y, W, b, layers, hyperparameters)
    model.extend([W, b])
    return model


def test_model(X, Y, W, b, layers, metric):
    X = X.T
    # Forward propagation for predictions
    Z, A = forward_prop(layers, W, b, X)

    m = metric_acquisition(A[-1], Y, metric)
    
    return m


def test_model_temp(X, Y, W, b, layers, metric):
    X = X.T
    # Forward propagation for predictions
    Z, A = forward_prop(layers, W, b, X)
    
    Y_pred = A[-1]

    plot_pred_vs_label(Y, Y_pred)

    m = metric_acquisition(A[-1], Y, metric)
    print(f"Metric value: ", m)


def blind_test(X, W, b, layers):
    X = X.T
    Z, A = forward_prop(layers, W, b, X)
    return A[-1].T


def train_and_evaluate(X_train, Y_train, X_test, Y_test, W, b, layers, hyperparameters):
    train_loss_data, test_loss_data = [], []
    train_metric_data, test_metric_data = [], []
    W_new, b_new = None, None

    for i in range(hyperparameters['epochs']):
        # Training phase
        Z_train, A_train = forward_prop(layers, W, b, X_train.T)
        dW, db = back_prop(layers, hyperparameters['err_fun'], Z_train, A_train, W, X_train.T, Y_train)
        W, b, W_new, b_new = update_params(len(layers), W, b, dW, db, hyperparameters['eta'],
                                           hyperparameters['alpha'], hyperparameters['lambd'], W_new, b_new)

        train_loss = mse(A_train[-1], Y_train)
        train_loss_data.append({'epoch': i, 'loss': train_loss})
        train_metric = metric_acquisition(A_train[-1], Y_train, hyperparameters['metric'])
        train_metric_data.append(train_metric)

        # Testing phase
        Z_test, A_test = forward_prop(layers, W, b, X_test.T)
        test_loss = mse(A_test[-1], Y_test)
        test_loss_data.append({'epoch': i, 'loss': test_loss})
        test_metric = metric_acquisition(A_test[-1], Y_test, hyperparameters['metric'])
        test_metric_data.append(test_metric)

    # Plotting
    epochs = range(hyperparameters['epochs'])
    plot_training_test_metrics(epochs, train_loss_data, test_loss_data, train_metric_data, test_metric_data, hyperparameters['metric'])

    return {
        'model': [W, b],
        'train_loss': train_loss_data[-1]['loss'],
        'test_loss': test_loss_data[-1]['loss'],
        'train_metric': train_metric_data[-1],
        'test_metric': test_metric_data[-1]
    }

