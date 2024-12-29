import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def one_hot_monk(X):
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
    loss_df = pd.DataFrame(data)
    loss_df.to_csv('loss_values.csv', index=False)

def loss_plot(path):
    loss_df = pd.read_csv(path)

    plt.plot(loss_df['epoch'], loss_df['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.show()

def metric_save(data, metric):
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
    plt.figure(figsize=(12, 5))

    # Plot training and test loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [data['loss'] for data in train_loss_data], label='Training')
    plt.plot(epochs, [data['loss'] for data in test_loss_data], label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()

    # Plot training and test metric
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_metric_data, label='Training')
    plt.plot(epochs, test_metric_data, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.title(f'Training and Test {metric} over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


