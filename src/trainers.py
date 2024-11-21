import numpy as np
import pandas as pd # maybe i can move this to another file
from error_fun import *
from metrics import *
from layer import Layer


# aggiungere un controllo per assicurarti che la dimensione di W[i] sia compatibile con A[i-1] per evitare
# errori difficili da tracciare

# define the forward propagaion
def forward_prop(layers, W, b, X):
    Z = [None] * len(layers)
    A = [None] * len(layers)

    #Z[0] = W[0].dot(X) + b[0]
    #A[0] = layers[0].activate(Z[0])

    for i in range(len(layers)):
        Z[i] = W[i].dot(A[i-1] if i > 0 else X) + b[i]
        A[i] = layers[i].activate(Z[i])
    
    return Z, A


# define the back propagation
def back_prop(layers, task, Z, A, W, X, Y):
    m = Y.size
    L = len(layers)
    dZ = [None] * L #maybe to delete and delete [] on dZ below
    dW = [None] * L
    db = [None] * L
    
    # compute dZ for the last layer
    dZ[-1] = error_comp(A[-1], Y, task)
    
    # loop backwards through layers to calculate gradients
    for i in reversed(range(L)):
        if i == L - 1:
            # output layer gradients
            dW[i] = 1 / m * dZ[i].dot(A[i-1].T)
            db[i] = 1 / m * np.sum(dZ[i], axis=1, keepdims=True)
        else:
            # hidden layer gradients
            dZ[i] = W[i + 1].T.dot(dZ[i+1]) * layers[i].activate_deriv(Z[i])
            dW[i] = 1 / m * dZ[i].dot(A[i-1].T if i > 0 else X.T)
            db[i] = 1 / m * np.sum(dZ[i], axis=1, keepdims=True)
    
    return dW, db


def update_params(num_layers, W, b, dW, db, eta, optimizer):
    if optimizer == 'none':
        for i in range(num_layers):
            W[i] = W[i] - eta * dW[i]
            b[i] = b[i] - eta * db[i]
        return W, b   


# gradient descend algorithm
def grad_descent(X, Y, W, b, layers, task, epochs, eta, optimizer):
    loss_data = []
    accuracy_data = []
    for i in range(epochs):
        Z, A = forward_prop(layers, W, b, X)

        acc = accuracy(A[-1], Y)
        accuracy_data.append(acc * 100)

        dW, db = back_prop(layers, task, Z, A, W, X, Y)
        W, b = update_params(len(layers), W, b, dW, db, eta, optimizer)

        loss = cross_entropy_loss(A[-1], Y) # TODO Change this fo MONK
        loss_data.append({'epoch': i, 'loss': loss})

    # TODO put this saving in a function
    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv('loss_values.csv', index=False)

    plot_accuracy(accuracy_data, epochs)

    return W, b


def train(algo, X, Y, W, b, layers, task, epochs, eta, optimizer):
    model = []
    if algo == 'gd':
        W, b = grad_descent(X, Y, W, b, layers, task, epochs, eta, optimizer)
        model.extend([W, b])
    return model

