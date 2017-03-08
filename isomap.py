import numpy as np
from numpy import linalg
from scipy.linalg import sqrtm as matrix_sqrt


def adjacency_lists(A, k):
    """
    Compute the adjacency lists for the k-nearest neighbors graph for the data
    A.
    """
    n = A.shape[0]
    J = np.empty((n, k), dtype=int)
    for i in range(n):
        # Take the i-th row of the adjacency matrix to be the indexes of the k
        # closest points to point i in ascending order.
        J[i,] = np.argsort(linalg.norm(A-A[i], axis=1))[1:k+1]
    return J


def graph_distance(J):
    """
    Compute the distance matrix for a graph from the adjacency lists in the
    array J.
    """
    (n, k) = J.shape
    # Compute distance matrix, initializing all entries to infinity.
    D = np.full((n, n), np.inf, dtype=np.float64)
    for i in range(n):       # For every point,
        D[i,i] = 0           # set distance to itself to 0,
        for j in range(k):   # and for all its neighbors,
            D[i,J[i,j]] = 1  # set distance to that neighbor to 1.
    return D


def compute_floyd_warshall(D):
    """
    Run the Floyd-Warshall algorithm on the partial distance matrix D to
    compute distances between all pairs of points.
    """
    n = D.shape[0]
    for k in range(n):
        for i in range(n):
            # Perform minimum distance checks for each row at a time (thanks to
            # numpy)---much faster.
            D[i,] = np.minimum(D[i,], D[i,k] + D[k,])
    return D


def centering_matrix(n):
    """
    Return a centering matrix of size n x n.
    """
    I = np.identity(n, dtype=np.float64)
    return I - np.full((n, n), 1./n, dtype=np.float64)


def gram_matrix(D):
    """
    Compute the Gram matrix G from the distance matrix D.
    """
    P = centering_matrix(D.shape[0])
    # Squared distance matrix (note that np.square is element-wise).
    D2 = np.square(D)
    # Compute the Gram matrix using the formula G = -PDP/2.
    G = (-0.5) * P.dot(D2).dot(P)
    # Forcibly symmetrize the resulting matrix (error should be less than
    # 1e-3 between entries before symmetrizing).
    G = np.minimum(G, G.T)
    return G


def sorted_eigenvectors(M):
    """
    Return the eigenvalues of the symmetric matrix M in descending order,
    together with the corresponding eigenvectors in the same order.
    """
    ws, vs = linalg.eigh(M)
    # Put eigenvalues and eigenvectors in descending (not ascending) order
    # (hence the `flipud` call).
    index = np.flipud(np.argsort(ws))
    # Return the eigenvalues and eigenvalues in the sorted order.
    ws = ws[index]
    vs = vs[:,index]
    return ws, vs


def mds_2D(D):
    """
    Perform multidimensional scaling starting from the distance matrix D.
    """
    # Compute the Gram matrix from the distance matrix.
    G = gram_matrix(D)
    # Compute the eigendecomposition of G.
    ws, vs = sorted_eigenvectors(G)
    # Take only the first two eigenvectors and eigenvalues, since we are
    # reducing to 2D.
    Q = vs[:,:2]
    L = np.diag(ws[:2])
    # Finally, compute the resulting vectors in 2D.
    points_2D = Q.dot(matrix_sqrt(L))
    # Split the result into x and y coordinates.
    x, y = np.hsplit(points_2D, [1])
    return (x, y)


def reduce_2D(A, k):
    """
    Reduce the data in A to a 2-dimensional subspace using isomap.
    :A: numpy.ndarray, n x m
    :k: int of number of nearest neighbors
    :return: numpy.ndarray, n x 2
    """
    return mds_2D(compute_floyd_warshall(graph_distance(adjacency_lists(A, k))))
