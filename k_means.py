from __future__ import division, print_function
from sklearn import datasets
import numpy as np

from unsupervisedLearning import KMeans
from data_helper.my_plot import Plot


if __name__ == "__main__":

    X, y = datasets.make_blobs()

    clf = KMeans(k=5)
    y_pred = clf.predict(X)

    # Plot 
    p = Plot()
    p.plot_in_2d(X, y_pred, title="K-Means Clusters")
    p.plot_in_2d(X, y, title="Actual Clusters")
