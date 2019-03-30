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
    def __init__(self, n_clusters=4, fuzziness=2, epsilon=0.0000000000001):

        self.n_clusters = n_clusters
        self.fuzziness = fuzziness
        self.epsilon = epsilon

    def fit(self, X):

        self.data = X

        # Create fuzzy partition matrix with random values
        self.fuzzy_matrix = np.random.randint(low=100, high=200, size=(self.data.shape[0], self.n_clusters))

        # Modify fuzzy partition matrix such that each row sums to 1
        self.fuzzy_matrix = self.fuzzy_matrix/self.fuzzy_matrix.sum(axis=1, keepdims=True)

        # Initial random centroid matrix
        self.centroids = np.random.randint(low=-4, high=4, size=(self.n_clusters, self.data.shape[1]))

        # Calculate SSE error
        fuzzy_matrix_powered = np.power(self.fuzzy_matrix, self.fuzziness)
        self.sse_error = 0
        for j in range(self.n_clusters):
            for i in range(self.data.shape[0]):
                self.sse_error += (fuzzy_matrix_powered[i][j] * np.power(np.linalg.norm(self.data[i]-self.centroids[j]), 2))

        fig = plt.figure()

        iteration_count = 1

        while True:

            # Print iteration number for clarity of progress
            stdout.write("\rIteration {}...".format(iteration_count))
            stdout.flush()
            iteration_count = iteration_count + 1

            ##################### UPDATE CENTROIDS #####################

            # Compute fuzzy_matrix's each element powered to fuzziness
            fuzzy_matrix_powered = np.power(self.fuzzy_matrix, self.fuzziness)

            # Divide each row of fuzzy_matrix_powered by the sum of the row
            fuzzy_matrix_powered = fuzzy_matrix_powered/fuzzy_matrix_powered.sum(axis=0, keepdims=True)

            # Compute new centroids (C = (W^p/sum(W^p))T * X)
            new_centroids = np.matmul(fuzzy_matrix_powered.T, self.data)


            ##################### UPDATE FUZZY MATRIX #####################

            # Update/Compute the new fuzzy matrix
            new_fuzzy_matrix = np.zeros(shape=(self.data.shape[0], self.n_clusters))
            for i in range(new_fuzzy_matrix.shape[0]):
                for j in range(new_fuzzy_matrix.shape[1]):
                    new_fuzzy_matrix[i][j] = 1./np.linalg.norm(self.data[i]-new_centroids[j])

            new_fuzzy_matrix = np.power(new_fuzzy_matrix, 2./(self.fuzziness-1))

            new_fuzzy_matrix = new_fuzzy_matrix/new_fuzzy_matrix.sum(axis=1, keepdims=True)


            #################### CALCULATE SSE ##################

            # Calculate new error, SSE
            new_fuzzy_matrix_powered = np.power(new_fuzzy_matrix, self.fuzziness)
            new_sse_error = 0.
            for j in range(self.n_clusters):
                for i in range(self.data.shape[0]):
                    new_sse_error += (new_fuzzy_matrix_powered[i][j] * np.power(np.linalg.norm(self.data[i]-new_centroids[j]), 2))

            # If change in SSE error is < epsilon, break
            if (self.sse_error - new_sse_error) < self.epsilon:
                break

            # Else, update centroids matrix and fuzzy matrix
            self.centroids = new_centroids.copy()
            self.fuzzy_matrix = new_fuzzy_matrix.copy()
            self.sse_error = new_sse_error

            # Plot the data
            plt.scatter(self.data[:,0], self.data[:,1], c='blue', marker='o')
            plt.scatter(self.centroids[:,0], self.centroids[:,1], c='red', marker='+')
            plt.pause(1)
            plt.close()


if __name__ == "__main__":

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Fuzzy c-means clustering')
    parser.add_argument('--n_clusters', '-k', type=int, default=4, help='No. of clusters')
    parser.add_argument('--fuzziness', '-m', type=int, default=2, help='Fuzziness parameter')
    parser.add_argument('--n_samples', '-n', type=int, default=500, help='No. of samples to generate')
    parser.add_argument('--epsilon', '-e', type=float, default=0.0000000000001, help='Stopping threshold')
    args = parser.parse_args()

    # Special case check for fuzziness = 1
    if (args.fuzziness == 1):
        print('Fuzziness value of 1 leads to divide by zero error')
        exit(1)

    # Generate data around 4 centers using make_blobs
    centers = [[-2,-2], [2,2], [2,-2], [-2,2]]
    
    X, _ = make_blobs(n_samples=args.n_samples, centers=centers, cluster_std=0.6)

    # Make an object of FuzzyCMeans class
    fcm = FuzzyCMeans(n_clusters=args.n_clusters, fuzziness=args.fuzziness, epsilon=args.epsilon)

    # Fit the data X
    fcm.fit(X)

    # Print centroids
    centroids = fcm.centroids

    print(centroids)

    # Plot the data
    plt.scatter(X[:,0], X[:,1], c='blue', marker='o')
    plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='+')
    plt.show()