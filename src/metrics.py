import numpy as np
import matplotlib.pyplot as plt


def accuracy(A, Y):

    pred_classes = np.argmax(A, axis=1)
    if Y.ndim == 1:
        true_classes = Y
    else:
        true_classes = np.argmax(Y, axis=1)
    return np.mean(pred_classes == true_classes)
    

def save_accuracy_plot(accuracy_data):
    plt.plot(range(len(accuracy_data)), accuracy_data, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.close()

