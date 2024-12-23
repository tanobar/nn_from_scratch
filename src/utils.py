import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def one_hot_vector(Y):
    one_hot = np.zeros((Y.size, Y.max() + 1))
    one_hot[np.arange(Y.size), Y.astype(int)] = 1
    return one_hot.T

def one_hot_monk(X):
    # Transpose data to process features row-wise
    X = X.T  # Dim: (num_examples, num_features)
   
    unique_values_per_feature = [3, 3, 2, 3, 4, 2]
    
    one_hot_encoded_features = []

    for col in range(X.shape[1]):
        feature = X[:, col]
        unique_values = unique_values_per_feature[col]
        
        one_hot = np.eye(unique_values)[feature.astype(int) - 1]  # One-hot encoding
        one_hot_encoded_features.append(one_hot)

    one_hot_encoded_matrix = np.concatenate(one_hot_encoded_features, axis=1)
    
    return one_hot_encoded_matrix.T

def loss_save(data):
    loss_df = pd.DataFrame(data)
    loss_df.to_csv('loss_values.csv', index=False)

def loss_plot(path):
    loss_df = pd.read_csv(path)

    plt.plot(loss_df['epoch'], loss_df['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.show()

def plot_pred_vs_label(Y, Y_pred):
    #plotting the graphic of targets vs predicions
    plt.plot(Y.flatten(), label="Target")
    plt.plot(Y_pred.flatten(), label="Model")
    plt.xlabel("Examples")
    plt.ylabel("Values")
    plt.title("Model Predictions vs. Target Values")
    plt.legend()
    plt.show()

def plot_accuracy(accuracy_data, epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), accuracy_data, label="Accuracy", color="blue", linewidth=2)
    plt.title("Accuracy vs Epochs", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)

    # enlight max accuracy
    max_acc = max(accuracy_data)
    max_epoch = accuracy_data.index(max_acc)
    plt.annotate(f"Max Accuracy: {max_acc:.2f}%", 
                 xy=(max_epoch, max_acc), 
                 xytext=(max_epoch + epochs * 0.1, max_acc - 5),
                 arrowprops=dict(facecolor='black', arrowstyle="->"),
                 fontsize=12)

    plt.show()

def plot_mee(mee_values):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mee_values)), mee_values, label="MEE", color="red", linewidth=2)
    plt.title("MEE vs Epochs", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("MEE", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)

    # enlight min mee
    min_mee = min(mee_values)
    min_epoch = mee_values.index(min_mee)
    plt.annotate(f"Min MEE: {min_mee:.2f}", 
                 xy=(min_epoch, min_mee), 
                 xytext=(min_epoch + len(mee_values) * 0.1, min_mee + 0.1),
                 arrowprops=dict(facecolor='black', arrowstyle="->"),
                 fontsize=12)

    plt.show()

def plot_metric(data, epochs, metric):
    print(metric)
    if metric == 'acc_bin':
        plot_accuracy(data, epochs)
        return
    if metric == 'mee':
        plot_mee(data)
        return
    else:
        raise ValueError('Invalid Metric')