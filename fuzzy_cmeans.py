"""
Author: Manohar Mukku
Date: 22 Mar 2019
Desc: Fuzzy c-means clustering
"""

from sys import stdout
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

class FuzzyCMeans:
    def __init__(self, n_clusters=4, fuzziness=2):

        self.n_clusters = n_clusters
        self.fuzziness = fuzziness

    def fit(self, X):

        self.data = X

        # Create fuzzy partition matrix with random values
        fuzzy_matrix = np.random.randint(low=100, high=200, size=(self.data.shape[0], self.n_clusters))

        # Modify fuzzy partition matrix such that each row sums to 1
        fuzzy_matrix = fuzzy_matrix/fuzzy_matrix.sum(axis=1, keepdims=True)

        # Initial empty centroid matrix
        self.centroids = np.zeros(shape=(self.n_clusters, self.data.shape[1]))

        iteration_count = 1

        while True:

            # Print iteration number for clarity of progress
            stdout.write("\rIteration {}...".format(iteration_count))
            stdout.flush()
            iteration_count = iteration_count + 1

            # Compute fuzzy_matrix's each element powered to fuzziness
            fuzzy_matrix_powered = np.power(fuzzy_matrix, self.fuzziness)

            # Divide each row of fuzzy_matrix_powered by the sum of the row
            fuzzy_matrix_powered = fuzzy_matrix_powered/fuzzy_matrix_powered.sum(axis=1, keepdims=True)

            # Compute centroids (C = (W^p/sum(W^p))T * X)
            new_centroids = np.matmul(fuzzy_matrix_powered.T, self.data)

            # Update the fuzzy_matrix
            for i in range(fuzzy_matrix.shape[0]):
                for j in range(fuzzy_matrix.shape[1]):
                    fuzzy_matrix[i][j] = 1./np.linalg.norm(self.data[i]-self.centroids[j])

            fuzzy_matrix = np.power(fuzzy_matrix, 2./(self.fuzziness-1))

            fuzzy_matrix = fuzzy_matrix/fuzzy_matrix.sum(axis=1, keepdims=True)

            # If centroids don't change, convergence reached, stop
            if (np.array_equal(self.centroids, new_centroids)):
                break

            # Else, update centroids matrix
            self.centroids = new_centroids.copy()


if __name__ == "__main__":

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Fuzzy c-means clustering')
    parser.add_argument('--n_clusters', '-k', type=int, default=4, help='No. of clusters')
    parser.add_argument('--fuzziness', '-m', type=int, default=2, help='Fuzziness parameter')
    parser.add_argument('--n_samples', '-n', type=int, default=500, help='No. of samples to generate')
    args = parser.parse_args()

    # Special case check for fuzziness = 1
    if (args.fuzziness == 1):
        print('Fuzziness value of 1 leads to divide by zero')
        exit(1)

    # Generate data around 4 centers using make_blobs
    centers = [[-2,-2], [2,2], [2,-2], [-2,2]]
    
    X, _ = make_blobs(n_samples=args.n_samples, centers=centers, cluster_std=0.6)

    # Make an object of FuzzyCMeans class
    fcm = FuzzyCMeans(n_clusters=args.n_clusters, fuzziness=args.fuzziness)

    # Fit the data X
    fcm.fit(X)

    # Print centroids
    centroids = fcm.centroids

    print(centroids)