"""
Author: Manohar Mukku
Date: 22 Mar 2019
Desc: Fuzzy c-means clustering
"""

import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

class FuzzyCMeans:
    def __init__(self, n_clusters=4, fuzziness=2):
        self.n_clusters = n_clusters
        self.fuzziness = fuzziness

    def fit(X):
        self.data = X

        fuzzy_partition_matrix = np.empty(shape=(self.data.shape[0], self.n_clusters), dtype=float)


if __name__ == "__main__":

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Fuzzy c-means clustering')
    parser.add_argument('--n_clusters', '-k', type=int, default=4, help='No. of clusters')
    parser.add_argument('--fuzziness', '-m', type=int, default=2, help='Fuzziness parameter')
    parser.add_argument('--n_samples', '-n', type=int, default=500, help='No. of samples to generate')
    args = parser.parse_args()

    # Generate data around 4 centers using make_blobs
    centers = [[-2,-2], [2,2], [2,-2], [-2,2]]
    
    X, _ = make_blobs(n_samples=args.n_samples, centers=centers, cluster_std=0.6)

    # Make an object of FuzzyCMeans class
    fcm = FuzzyCMeans(n_clusters=args.n_clusters, fuzziness=args.fuzziness)

    # Fit the data X
    fcm.fit(X)