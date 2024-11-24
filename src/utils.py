import numpy as np

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
