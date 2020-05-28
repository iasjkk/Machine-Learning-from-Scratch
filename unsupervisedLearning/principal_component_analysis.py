from __future__ import print_function, division
import numpy as np
from utils.data_operation import calculate_covariance_matrix


class PCA():
    """
    Principal Component Analysis(PCA) is one of the most popular 
    linear dimension reduction. Sometimes, it is used alone and 
    sometimes as a starting solution for other dimension reduction
    methods. PCA is a projection based method which transforms 
    the data by projecting it onto a set of orthogonal axes. 
    It remove the common texture from the features i.e, correlation between features
    and maximises the variance along the feature axes.
    
    I have Used the PCA for image recognition in my bachelor project. 
    It extracts discriminative features from images.
    """
    def transform(self, X, n_components):
        """ Fit the dataset to the number of principal components specified in the
        constructor and return the transformed dataset """
        covariance_matrix = calculate_covariance_matrix(X)

        # Where (eigenvector[:,0] corresponds to eigenvalue[0])
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]

        # Project the data onto principal components
        X_transformed = X.dot(eigenvectors)

        return X_transformed
