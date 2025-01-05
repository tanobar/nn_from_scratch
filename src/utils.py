import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def one_hot_monk(X):
    """
    Perform one-hot encoding on the given dataset for the MONK problem.
    Parameters:
    X (numpy.ndarray): A 2D array where each row represents a feature and each column represents an example.
                       The shape of X should be (num_features, num_examples).
    Returns:
    numpy.ndarray: A 2D array where each row represents an example and each column represents a one-hot encoded feature.
                   The shape of the returned array will be (num_examples, sum(unique_values_per_feature)).
    """
    # Transpose data to process examples row-wise
    X = X.T  # Dim: (num_features, num_examples)
   
    unique_values_per_feature = [3, 3, 2, 3, 4, 2]
    
    one_hot_encoded_features = []

    for row in range(X.shape[0]):
        feature = X[row, :]
        unique_values = unique_values_per_feature[row]
        
        one_hot = np.eye(unique_values)[feature.astype(int) - 1]  # One-hot encoding
        one_hot_encoded_features.append(one_hot)

    one_hot_encoded_matrix = np.concatenate(one_hot_encoded_features, axis=1)
    
    return one_hot_encoded_matrix

def loss_save(data):
    """
    Save loss data to a CSV file.

    Parameters:
    data (list or dict): The loss data to be saved. It can be a list of dictionaries or a dictionary of lists.

    Returns:
    None
    """
    loss_df = pd.DataFrame(data)
    loss_df.to_csv('loss_values.csv', index=False)

def loss_plot(path):
    """
    Plots the loss over epochs from a CSV file.
    Parameters:
    path (str): The file path to the CSV file containing the loss data. 
                The CSV file should have columns 'epoch' and 'loss'.
    Returns:
    None
    """
    loss_df = pd.read_csv(path)

    plt.plot(loss_df['epoch'], loss_df['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.show()

def metric_save(data, metric):
    """
    Save metric data to a CSV file.

    Parameters:
    data (list or array-like): The data to be saved.
    metric (str): The name of the metric, which will be used as the column name in the CSV file.

    Returns:
    None
    """
    metric_df = pd.DataFrame(data, columns=[metric])
    metric_df.to_csv(f'{metric}_values.csv', index=False)

def plot_pred_vs_label(Y, Y_pred):
    #plotting the graphic of targets vs predicions
    plt.plot(Y.flatten(), label="Target")
    plt.plot(Y_pred.flatten(), label="Model")
    plt.xlabel("Examples")
    plt.ylabel("Values")
    plt.title("Model Predictions vs. Target Values")
    plt.legend()
    plt.show()

def plot_accuracy(path):
    """
    Plots the accuracy over epochs from a CSV file.
    Parameters:
    path (str): The file path to the CSV file containing accuracy data. 
                The CSV file should have a column named 'acc_bin' with accuracy values.
    The function reads the accuracy data from the CSV file, plots it against the number of epochs,
    and highlights the maximum accuracy achieved during the training process.
    The plot includes:
    - A line plot of accuracy vs. epochs.
    - A title "Accuracy vs Epochs".
    - X-axis labeled as "Epochs".
    - Y-axis labeled as "Accuracy (%)".
    - A grid for better readability.
    - A legend indicating the accuracy line.
    - An annotation pointing to the maximum accuracy value on the plot.
    The plot is displayed using matplotlib's `plt.show()` function.
    """
    accuracy_df = pd.read_csv(path)
    accuracy_data = accuracy_df['acc_bin'].tolist()
    epochs = len(accuracy_data)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), accuracy_data, label="Accuracy", color="blue", linewidth=2)
    plt.title("Accuracy vs Epochs", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)

    # highlight max accuracy
    max_acc = max(accuracy_data)
    max_epoch = accuracy_data.index(max_acc)
    plt.annotate(f"Max Accuracy: {max_acc:.2f}%", 
                 xy=(max_epoch, max_acc), 
                 xytext=(max_epoch + epochs * 0.1, max_acc - 5),
                 arrowprops=dict(facecolor='black', arrowstyle="->"),
                 fontsize=12)

    plt.show()

def plot_mee(path):
    """
    Plots the Mean Euclidean Error (MEE) over epochs from a CSV file.
    Parameters:
    path (str): The file path to the CSV file containing the MEE data. The CSV file should have a column named 'mee'.
    The function reads the MEE data from the CSV file, plots it against the number of epochs, and highlights the minimum MEE value on the plot.
    """
    mee_df = pd.read_csv(path)
    mee_data = mee_df['mee'].tolist()
    epochs = len(mee_data)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), mee_data, label="MEE", color="red", linewidth=2)
    plt.title("MEE vs Epochs", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("MEE", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)

    # highlight min mee
    min_mee = min(mee_data)
    min_epoch = mee_data.index(min_mee)
    plt.annotate(f"Min MEE: {min_mee:.2f}", 
                 xy=(min_epoch, min_mee), 
                 xytext=(min_epoch + epochs * 0.1, min_mee + 0.1),
                 arrowprops=dict(facecolor='black', arrowstyle="->"),
                 fontsize=12)

    plt.show()

def plot_training_test_metrics(epochs, train_loss_data, test_loss_data, train_metric_data, test_metric_data, metric):
    """
    Plots the training and test loss and a specified metric over epochs.
    Parameters:
    epochs (list): A list of epoch numbers.
    train_loss_data (list): A list of dictionaries containing training loss data for each epoch.
    test_loss_data (list): A list of dictionaries containing test loss data for each epoch.
    train_metric_data (list): A list of training metric values for each epoch.
    test_metric_data (list): A list of test metric values for each epoch.
    metric (str): The name of the metric to be plotted (e.g., 'Accuracy', 'Precision').
    Returns:
    None
    """
    plt.figure(figsize=(12, 5))

    # Plot training and test loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [data['loss'] for data in train_loss_data], label='Training')
    plt.plot(epochs, [data['loss'] for data in test_loss_data], label='Test', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()

    # Plot training and test metric
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_metric_data, label='Training')
    plt.plot(epochs, test_metric_data, label='Test', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.title(f'Training and Test {metric} over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


