from __future__ import print_function, division
import numpy as np
from utils.data_operation import euclidean_distance

class KNN():
    """ K Nearest Neighbors classifier has no model other than storing the entire dataset, so there is no learning required..

    Parameters:
    -----------
    k: int
        The no. of closest neighbors that will determine the class of the 
        sample that we wish to predict.
    """
    def __init__(self, k=5):
        self.k = k

    def _vote(self, neighbor_labels):
        """ Return the most common class among the neighbor samples """
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()

    def predict(self, X_test, X_train, y_train):
        y_pred = np.empty(X_test.shape[0])
        # Determine the label of each test sample
        for i, test_sample in enumerate(X_test):
            # Sort the training samples by their distance to the test sample and get the K nearest
            idx = np.argsort([euclidean_distance(test_sample, x) for x in X_train])[:self.k]
            # Extract the labels of the K nearest neighboring training samples
            k_nearest_neighbors = np.array([y_train[i] for i in idx])
            # Majority label will be assigned to the test sample
            y_pred[i] = self._vote(k_nearest_neighbors)

        return y_pred
        