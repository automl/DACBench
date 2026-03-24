"""Exploration features from Exploring Exploration in Bayesian Optimization [Papenmeier et al., 2025]."""  # noqa: N999

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.special import digamma


# TODO add normalized TSD
def exploration_tsd(X: np.ndarray) -> np.ndarray:
    """Movement of observation center over time.

    Parameters
    ----------
    X : np.ndarray, shape (T, D)

    Returns:
    -------
    np.ndarray, shape (T, )
    """
    T, _D = X.shape
    tsd_solution = np.zeros(shape=(T,), dtype=np.float32)

    # Initialize with the first point's TSP solution
    current_path = [0, 0]
    cumulative_distance = 0.0
    tsd_solution[0] = cumulative_distance

    # Precompute the distance matrix
    dist_matrix = squareform(pdist(X)).astype(np.float32)

    # Calculate distances incrementally
    for t in range(1, T):
        # Update path by finding the best place to insert the new point
        best_distance_increase = float("inf")
        best_insertion_index = -1
        # Update distance matrix to include the new point
        inner_dist_matrix = dist_matrix[: t + 1, : t + 1]

        # Try inserting the new point in each position in the current path
        for i in range(len(current_path) - 1):
            # Calculate distance if new point were inserted between path[i] and path[i+1]
            dist_increase = (
                inner_dist_matrix[current_path[i], t]
                + inner_dist_matrix[t, current_path[i + 1]]
                - inner_dist_matrix[current_path[i], current_path[i + 1]]
            )
            if dist_increase < best_distance_increase:
                best_distance_increase = dist_increase
                best_insertion_index = i + 1

        # Insert the new point at the best position found
        if best_insertion_index != -1:
            current_path.insert(best_insertion_index, t)
        cumulative_distance += best_distance_increase
        tsd_solution[t] = cumulative_distance

    return tsd_solution


def knn_entropy(X: np.ndarray, k: int = 3) -> float:
    """Estimate the Shannon entropy of a dataset using the k-nearest neighbors method.

    Parameters
    ----------
    X : numpy.ndarray
        The N x D array of data points.
    k : int
        Number of neighbors to use in the k-NN estimation.

    Returns:
    -------
    float
        The estimated entropy.
    """
    N = X.shape[0]
    D = X.shape[1]

    tree = cKDTree(X)
    distances = tree.query(X, k=k + 1, p=2)[
        0
    ]  # k+1 because the point itself is included
    nn_distances = distances[:, k]  # k-th nearest neighbor distance
    avg_log_dist = np.mean(
        np.log(nn_distances + 1e-10)
    )  # Add small value to avoid log(0)

    # Volume of the D-dimensional hypersphere
    V = np.pi ** (D / 2) / math.gamma(D / 2 + 1)

    # Calculate the entropy
    return digamma(N) - digamma(k) + np.log(V) + D * avg_log_dist


def exploration_entropy(X: np.ndarray) -> np.ndarray:
    """Calculate the empirical Shannon entropy over cumulative observation points at each time step,
    dynamically setting k based on the sample size.

    Parameters
    ----------
    X : numpy.ndarray
        The T x D array where each row is a data point in [0, 1]^D.

    Returns:
    -------
    numpy.ndarray
        An array of entropy values for each time step.
    """
    T = X.shape[0]
    D = X.shape[1]
    # Due to singularity, the first D points are ignored
    entropies = np.zeros(T - D, dtype=np.float32)

    for eidx, t in enumerate(range(D, T)):
        # Dynamically set k as the square root of current sample size
        k = max(1, int(np.log(t + 1)))

        # Estimate entropy using k-NN on cumulative data
        entropies[eidx] = knn_entropy(X[:t], k=k)

    return entropies
