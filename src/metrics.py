import numpy as np
import matplotlib.pyplot as plt


def accuracy_bin_classification(A, Y):
    predicted_classes = (A >= 0.5).astype(int)
    return np.mean(predicted_classes == Y)

def accuracy(A, Y): # TODO change and modularize
    return accuracy_bin_classification(A, Y)
    

def plot_accuracy(accuracy_data, epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), accuracy_data, label="Accuracy", color="blue", linewidth=2)
    plt.title("Accuracy vs Epochs", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)

    # Evidenzia il massimo valore di accuracy
    max_acc = max(accuracy_data)
    max_epoch = accuracy_data.index(max_acc)
    plt.annotate(f"Max Accuracy: {max_acc:.2f}%", 
                 xy=(max_epoch, max_acc), 
                 xytext=(max_epoch + epochs * 0.1, max_acc - 5),
                 arrowprops=dict(facecolor='black', arrowstyle="->"),
                 fontsize=12)

    plt.show()

