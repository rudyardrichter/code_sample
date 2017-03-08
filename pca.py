import numpy as np
from numpy import linalg


def center(A):
    """
    ``Center'' the data around the mean.
    :A: numpy.ndarray to be centered
    :return: None (side effects only)
    """
    A -= A.mean(axis=0)
    return


def cov(A):
    """
    Return the sample covariance matrix for the data A.
    :A: numpy.ndarray, n x m
    :return: numpy.ndarray, m x m
    """
    return np.cov(A, rowvar=False)


def reduce_2D(A):
    """
    Reduce the data in A to a 2-dimensional subspace with the basis being the
    first two principal components.
    :A: numpy.ndarray, n x m
    :return: numpy.ndarray, n x 2
    """
    center(A)
    C = cov(A)
    # Compute all the eigenvalues ws and all the eigenvectors vs of the
    # covariance matrix. (Covariance matrix is always symmetric, hence use of
    # `eigh` for optimization.)
    ws, vs = linalg.eigh(C)
    # Argsort the eigenvalues in descending order, then take the first two
    # eigenvectors corresponding to the largest eigenvalues.
    principals = vs[np.argsort(-ws)][:2]
    # Project the data into the 2-dimensional subspace described by the
    # principal eigenvectors.
    points_2D = np.dot(principals, A.T).T
    x, y = np.hsplit(points_2D, [1])
    return (x, y)
