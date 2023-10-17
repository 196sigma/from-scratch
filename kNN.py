# Implement k-Nearest Neighbors algorithm here
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k=3, X=None, y=None):
        """
        Parameters
        ----------
        k : int
            Number of nearest neighbors to consider.
        X : np.ndarray
            Training data.
        y : np.ndarray
            Training labels.
        """
        self.k = k
        self.X = X
        self.y = y
    
    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.

        Parameters
        ----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Training labels.
        """
        self.X = X
        self.y = y

    def predict(self, X_new):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X_new : np.ndarray
            Data to classify.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        y_pred = [self._predict(x) for x in X_new]
        return np.array(y_pred)
    
    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [distance(x, x_train) for x_train in self.X]
        
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y[i] for i in k_idx]  

        # take a vote among the k neighbors
        most_common = Counter(k_neighbor_labels).most_common(1)
    
    def score(self, X, y):
        pass

    def distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    