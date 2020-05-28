from __future__ import print_function, division
import numpy as np

# helper function
from data_helper.pre_processing import normalize
from data_helper.eval_function import euclidean_distance
from data_helper.my_plot import Plot


class KMeans():
    """Ais one of the simplest unsupervised learning algorithms that 
    solve the well known clustering problem. The procedure follows a 
    simple and easy way to classify a given data set through a certain 
    number of clusters (assume k clusters) fixed a priori.
    
    1. Place K points into the space represented by the objects that 
    are being clustered. These points represent initial group centroids.
    2. Assign each object to the group that has the closest centroid.
    3. When all objects have been assigned, recalculate the positions 
    of the K centroids.
    4. Repeat Steps 2 and 3 until the centroids no longer move. 
    This produces a separation of the objects into groups from which 
    the metric to be minimized can be calculated."""


    def __init__(self, k=2, max_iterations=800):
        self.k = k
        self.max_iterations = max_iterations

    def initialise_random_centroids(self, X):
        """ Initialize the k centroids as k random samples of X"""
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    def closest_centroid(self, sample, centroids):
        """ Return the index of the closest centroid to the sample """
        closest_i = 0
        closest_dist = float('inf')
        for i, centroid in enumerate(centroids):
            distance = euclidean_distance(sample, centroid)
            if distance < closest_dist:
                closest_i = i
                closest_dist = distance
        return closest_i

    def create_clusters(self, centroids, X):
        """ Assign the samples to the closest centroids to create clusters """
        n_samples = np.shape(X)[0]
        clusters = [[] for i in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self.closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    def eval_centroids(self, clusters, X):
        """ Calculate new centroids as the means of the samples in each cluster  """
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    def assigned_cluster_labels(self, clusters, X):
        # It predicts single label for each sample 
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    def predict(self, X):
        """ Do K-Means clustering and return cluster indices """

        # Initialize centroids as k random samples from X
        centroids = self.initialise_random_centroids(X)

        # Early stop convergence
        for iter in range(self.max_iterations):
            # Assign samples to closest centroids (create clusters)
            clusters = self.create_clusters(centroids, X)
            # Save current centroids for convergence check
            prev_centroids = centroids
            # Calculate new centroids from the clusters
            centroids = self.eval_centroids(clusters, X)
            print(centroids)
            # If no centroids have changed => convergence
            diff = centroids - prev_centroids
            if not diff.any():
                break

        return self.assigned_cluster_labels(clusters, X)

