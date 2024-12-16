import numpy as np
import pandas as pd # maybe i can move this to another file
from error_fun import *
from metrics import *
from layer import Layer


def forward_prop(layers, W, b, X):
    Z, A = [None] * len(layers), [None] * len(layers)

    for i in range(len(layers)):
        Z[i] = W[i].dot(A[i-1] if i > 0 else X) + b[i]
        A[i] = layers[i].activate(Z[i])
    
    return Z, A


def back_prop(layers, err_fun, Z, A, W, X, Y):
    L = len(layers)
    dZ, dW, db = [None] * L, [None] * L, [None] * L # maybe to delete and delete [] on dZ below
    
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


def update_params(num_layers, W, b, dW, db, eta, momentum, alpha, lambd, W_new, b_new):
    if momentum:
        if W_new is None:
            W_new = [np.zeros_like(W[i]) for i in range(num_layers)]
        if b_new is None:
            b_new = [np.zeros_like(b[i]) for i in range(num_layers)]

        for i in range(num_layers):
            # momentum
            W_new[i] = eta * dW[i] + alpha * W_new[i]
            b_new[i] = eta * db[i] + alpha * b_new[i]

            if lambd > 0:
                W[i] = W[i] - W_new[i] - eta * lambd * W[i]
            else:
                W[i] = W[i] - W_new[i]
            b[i] = b[i] - b_new[i]

    else:
        for i in range(num_layers):
            if lambd > 0:
                W[i] = W[i] - eta * dW[i] - eta * lambd * W[i]
            else:
                W[i] = W[i] - eta * dW[i]
            b[i] = b[i] - eta * db[i]

    return W, b, W_new, b_new


# gradient descend algorithm
def grad_descent(X, Y, W, b, layers, hyperparameters):
    loss_data, accuracy_data = [], []
    W_new, b_new = None, None
    for i in range(hyperparameters['epochs']):
        Z, A = forward_prop(layers, W, b, X)

        acc = accuracy(A[-1], Y)
        accuracy_data.append(acc * 100)

        dW, db = back_prop(layers, hyperparameters['err_fun'], Z, A, W, X, Y)
        W, b, W_new, b_new = update_params(len(layers), W, b, dW, db, hyperparameters['eta'],
                                            hyperparameters['momentum'], hyperparameters['alpha'],
                                            hyperparameters['lambd'], W_new, b_new)

        loss = mse(A[-1], Y)
        loss_data.append({'epoch': i, 'loss': loss})

    # TODO put this saving in a function (maybe)
    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv('loss_values.csv', index=False)

    plot_accuracy(accuracy_data, hyperparameters['epochs'])

    return W, b


def train_model(X, Y, W, b, layers, hyperparameters):
    model = []
    if hyperparameters['optimizer'] == 'gd':
        W, b = grad_descent(X, Y, W, b, layers, hyperparameters)
        model.extend([W, b])
    return model


def test_model(W, b, layers, X, Y):
    
    # Forward propagation for predictions
    Z, A = forward_prop(layers, W, b, X)
    
    Y_pred = A[-1]

    acc = accuracy(A[-1], Y) * 100
    print(f"Test Accuracy: " , acc)

    #plotting the graphic of targets vs predicions
    plt.plot(Y.flatten(), label="Target")
    plt.plot(Y_pred.flatten(), label="Model")
    plt.xlabel("Examples")
    plt.ylabel("Values")
    plt.title("Model Predictions vs. Target Values")
    plt.legend()
    plt.show()

