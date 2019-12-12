import warnings

import numpy as np
from sklearn.decomposition import NMF


def whiten(X, num_components=None, center=True, rowvar=True):
    """Whiten the data in matrix X using PCA decomposition.

    The data corresponds to n samples of a p-dimensional random vector. The shape
    of the matrix can be either (n, p) if each row is considered to be a sample or
    (p, n) if each column is considered to be a sample. How to read the matrix entries
    is specified by the rowvar parameter. Before whitening, a dimensionality reduction
    step can be applied to the data to reduce the p dimensions of each sample to
    num_components dimensions. If num_components is None, the number of dimensions kept
    is the maximum possible (nº of non-zero eigenvalues). For example, if X is full rank
    (rank(X) = min(n, p)), then num_components = p if p < n, and num_components = n-1
    if p >= n.

    Args:
        X: Data matrix.
        num_components: Number of PCA dimensions of the whitened samples.
        center: Whether to center the samples or not (zero-mean whitened samples).
        rowvar: Whether each row of X corresponds to one of the p variables or not.

    Returns:
        (Z, V): The whitened data matrix and the whitening matrix.

    """
    r = num_components

    if rowvar:
        X = X.transpose()

    # Data matrix contains n observations of a p-dimensional vector
    # Each observation is a row of X
    n, p = X.shape

    # Arbitrary (but sensible) choice. In any case, we remove the eigenvectors of 0 eigenvalue later
    if r is None:
        r = min(n, p)

    # Compute the mean of the observations (p-dimensional vector)
    mu = np.mean(X, axis=0)

    # If p > n compute the eigenvectors efficiently
    if p > n:
        # n x n matrix
        M = np.matmul((X-mu), (X-mu).transpose())

        # Eigenvector decomposition
        vals, vecs = np.linalg.eig(M)
        vals, vecs = vals.real, vecs.real

        # Sort the eigenvectors by "importance" and get the first r
        pairs = sorted([(vals[i], vecs[:, i]) for i in range(len(vals))], key=lambda x: x[0], reverse=True)
        pairs = [p for p in pairs if abs(p[0]) > 1e-10]  # Remove the eigenvectors of 0 eigenvalue
        pairs = pairs[:r]

        # nxr matrix of eigenvectors (each column is an n-dimensional eigenvector)
        E = np.array([p[1] for p in pairs]).transpose()

        # pxr matrix of the first r eigenvectors of the covariance of X
        # Note that we normalize!
        E = np.matmul((X-mu).transpose(), E)
        E /= np.linalg.norm(E, axis=0)

        # Eigenvalues of cov(X) to the -1/2
        # Note that we rescale the eigenvalues of M to get the eigenvalues of cov(X)!
        diag = np.array([1/np.sqrt(p[0]/(n-1)) for p in pairs])

    else:
        # p x p matrix
        C = np.cov(X, rowvar=False)

        # Eigenvector decomposition
        vals, vecs = np.linalg.eig(C)
        vals, vecs = vals.real, vecs.real

        # Sort the eigenvectors by "importance" and get the first r
        pairs = sorted([(vals[i], vecs[:, i]) for i in range(len(vals))], key=lambda x: x[0], reverse=True)
        pairs = [p for p in pairs if abs(p[0]) > 1e-10]  # Remove the eigenvectors of 0 eigenvalue
        pairs = pairs[:r]

        # pxr matrix of the first r eigenvectors of the covariance of X
        E = np.array([p[1] for p in pairs]).transpose()

        # Eigenvalues of cov(X) to the -1/2
        diag = np.array([1/np.sqrt(p[0]) for p in pairs])

    # Warn that the specified number of components is larger
    # than the number of non-zero eigenvalues.
    if num_components is not None:
        if num_components > len(pairs):
            warnings.warn(
                'The desired number of components (%d) is larger than the actual dimension'
                ' of the PCA subespace (%d)' % (num_components, len(pairs))
            )

    # Center and whiten the data
    if center:
        X = X - mu

    # Whitening matrix
    V = E * diag

    # White data
    Z = np.matmul(X, V)

    if rowvar:
        Z = Z.transpose()

    # Since X is assumed to be (n, p) through the computations, the current
    # whitening matrix V is in fact the transpose of the actual whitening matrix.
    # Observation: If z = V * x for random column vectors x, z, then Z = X * V
    # for the (n, p) and (n, r) matrices X, Z of observations of x, z.
    V = V.transpose()

    return Z, V


def nica(X, num_sources, lr=0.03, max_iter=5000, tol=1e-8, rowvar=True):
    """Compute the non-negative independent components of the linear generative model x = A * s.

    Here, x is a p-dimensional observable random vector and s is the latent random vector
    of length num_sources, whose components are statistically independent and non-negative.
    The matrix X is assumed to hold n samples of x, stacked in rows (shape(X) = (n, p)) or
    columns (shape(X) = (p, n)), which can be specified by the rowvar parameter. In practice,
    if shape(X) = (p, n) (resp. shape(X) = (n, p)) this function solves X = A * S
    (resp. X = S.T * A.T) both for S and A, where A is the so-called mixing matrix, with shape
    (p, num_sources), and S is a (num_sources, n) matrix which contains n samples of the latent
    source vector, stacked in columns.

    This function implements the method presented in:
    `Blind Separation of Positive Sources by Globally Convergent Gradient Search´
    (https://core.ac.uk/download/pdf/76988305.pdf)

    Args:
        X: Data matrix.
        num_sources: Dimension of s. Number of latent random variables.
        lr: Learning rate of gradient descent.
        max_iter: Maximum number of iterations of gradient descent.
        tol: Tolerance on update at each iteration.
        rowvar: Whether each row of X corresponds to one of the p variables or not.

    Returns:
        (S, A) if rowvar == True.
        (S.T, A) if rowvar == False.

    """
    # Whiten the data
    Z, V = whiten(X, num_sources, center=False, rowvar=rowvar)

    if num_sources > V.shape[0]:
        warnings.warn(
            'The desired number of sources (%d) is larger than the actual dimension'
            ' of the whitened observable random vector (%d). The number of sources'
            ' will be set to %d' % (num_sources, V.shape[0], V.shape[0])
        )
        num_sources = V.shape[0]

    # We assume rowvar is True throughout the algorithm
    if not rowvar:
        Z = Z.transpose()

    # Initialize W
    W = np.eye(num_sources)

    for i in range(max_iter):
        W0 = W

        # Compute gradient
        Y = np.matmul(W, Z)
        f = np.minimum(0, Y)
        f_Y = np.matmul(f, Y.transpose())
        E = (f_Y - f_Y.transpose()) / Y.shape[1]

        # Gradient descent
        W -= lr * np.matmul(E, W)

        # Symmetric orthogonalization
        M = np.matmul(W, W.transpose())
        vals, vecs = np.linalg.eig(M)
        vals, vecs = vals.real, vecs.real

        W_sqrt = vecs / np.sqrt(vals)
        W_sqrt = np.matmul(W_sqrt, vecs.transpose())
        W = np.matmul(W_sqrt, W)

        if np.linalg.norm(W - W0) < tol:
            break

    # Independent sources (up to an unknown permutation y = Q * s)
    Y = np.matmul(W, Z)

    # Compute the mixing matrix A' = A * Q.T
    # (which is A up to a permutation of its columns)
    # from the identity y = Q * s = W * V * A * s.
    # It then holds x = A * s = A * Q.T * y = A' * y.
    # Note: A' is computed as the right Moore-Penrose
    # inverse of W * V, but A' may not be unique since
    # in general p != num_sources and any right inverse
    # could be taken as A'.
    WV = np.matmul(W, V)
    WV_ = np.matmul(WV, WV.transpose())
    WV_ = np.linalg.inv(WV_)
    WV_ = np.matmul(WV.transpose(), WV_)

    if not rowvar:
        Y = Y.transpose()

    return Y, WV_


def nica_nmf(X, num_components, lr=0.03, max_iter=5000, tol=1e-8, rowvar=True):
    """Non-negative matrix factorization with non-negative ICA (NICA) initialization.

    Under the linear generative model x = A * s, where x is a p-dimensional observable
    random vector, s is the latent non-negative random vector of length num_components and
    A is a fixed (but unknown) non-negative matrix, this function tries to determine both s
    and A. The data matrix X is assumed to hold n samples of x, stacked in rows (shape(X) = (n, p))
    or columns (shape(X) = (p, n)), which can be specified by the rowvar parameter. In practice,
    if shape(X) = (p, n) (resp. shape(X) = (n, p)) this function solves X = A * S
    (resp. X = S.T * A.T) both for S and A, where A is the so-called mixing matrix, with shape
    (p, num_sources), and S is a (num_sources, n) matrix which contains n samples of the latent
    source vector, stacked in columns.

    The non-uniqueness (non-convexity) property of NMF implies that the solution depends on the
    initial factor matrices. This function implements the idea presented in:
    `Efficient initialization for nonnegative matrix factorization based on nonnegative independent component analysis´
    (https://ieeexplore.ieee.org/document/7602947)
    which suggests that a good initialization is based on the factorization given by non-negative ICA.

    Args:
        X: Data matrix.
        num_components: Dimension of s. Number of latent random variables.
        lr: Learning rate of gradient descent for NICA.
        max_iter: Maximum number of iterations of gradient descent for NICA.
        tol: Tolerance on update at each iteration for NICA.
        rowvar: Whether each row of X corresponds to one of the p variables or not.

    Returns:
        (S, A) if rowvar == True.
        (S.T, A) if rowvar == False.

    """
    S, A = nica(X, num_components, lr, max_iter, tol, rowvar)

    # We assume rowvar is True throughout the algorithm
    if not rowvar:
        X = X.transpose()
        S = S.transpose()

    # Initial NMF factorization: X = F0 * G0
    F0 = np.abs(A)
    G0 = np.abs(S)

    W0 = G0.transpose().copy()  # Make array C-contiguous
    H0 = F0.transpose()

    nmf = NMF(n_components=num_components, init='custom')

    W = nmf.fit_transform(X.transpose(), W=W0, H=H0)
    H = nmf.components_

    A = H.transpose()
    S = W.transpose()

    if not rowvar:
        S = S.transpose()

    return S, A
