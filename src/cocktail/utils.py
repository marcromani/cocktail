import random

import numpy as np


def _lstsq_vector(A, b, constraints=None):
    """Minimize || A*x - b || subject to equality constraints x_i = c_i.

    Let A be a matrix of shape (m, n) and b a vector of length m. This function
    solves the minimization problem || A*x - b || for x, subject to 0 <= r <= n
    equality constraints x_i = c_i. The n entries of the vector of constraints
    must be either NaN (if there is no constraint for that entry) or a scalar.

    Args:
        A: Coefficient matrix.
        b: Dependent variable.
        constraints: Constraint vector.

    Returns:
        x: The minimizer of the problem.

    """
    if constraints is None:
        return np.linalg.lstsq(A, b, rcond=None)[0]

    # Indices of the constraints on x
    indices = np.nonzero(np.isfinite(constraints))[0]
    # Values of the constraints on x
    c = constraints[np.isfinite(constraints)]
    # Number of constraints
    r = c.size

    n = A.shape[1]

    # Matrix P.T projects (x_1, ..., x_n) to (x_i1, ..., x_ir)
    # where ij are the indices of the constraints on x
    P = np.zeros((n, r))
    for i, x in enumerate(indices):
        P[x][i] = 1

    A00 = np.matmul(A.T, A)
    A01 = P
    A10 = P.T
    A11 = np.zeros(2*(P.shape[1],))

    # Augmented A
    A_ = np.block([[A00, A01], [A10, A11]])

    b0 = np.matmul(A.T, b)
    b1 = c

    # Augmented b
    b_ = np.block([b0, b1])

    # Solve the augmented system
    x = np.linalg.lstsq(A_, b_, rcond=None)[0]

    return x[:n]


def _lstsq_matrix(X, Y, constraints=None):
    """Minimize || A*X - Y || for A, subject to equality constraints a_ij = c_ij.

    Let X, Y be matrices of shapes (n, k), (m, k) respectively, so that A is a
    matrix of shape (m, n). This function solves the minimization problem
    || A*X - Y || for A, subject to 0 <= r <= m*n equality constraints a_ij = c_ij.
    The entries of the (m, n) matrix of constraints must be either NaN (if there
    is no constraint for that entry) or a scalar.

    Args:
        X: Input matrix.
        Y: Output matrix.
        constraints: Constraint matrix.

    Returns:
        A: The minimizer of the problem.

    """
    if constraints is None:
        return np.linalg.lstsq(X.T, Y.T, rcond=None)[0].T

    A = np.empty((Y.shape[0], X.shape[0]))

    for i in range(Y.shape[0]):
        A[i] = _lstsq_vector(X.T, Y[i], constraints[i])

    return A


def lstsq(a, b, constraints=None):
    """Minimize || a*x - b || for x, subject to equality constraints on the elements of x.

    Find the minimum of the function L(x) = || a*x - b ||, where a is a matrix of shape
    (m, n) and b is either a vector of length m or a matrix of shape (m, k). In the first
    case, the solution x is a vector of length n, in the latter, it is a matrix of shape
    (n, k). If x is a vector (resp. a matrix), 0 <= r <= n (resp. 0 <= r <= n*k) constraints
    of the form x_i = c_i (resp. x_ij = c_ij) can be provided. In particular, the entries
    of the n-vector (resp. (n, k)-matrix) of constraints must be either NaN (if there is no
    constraint for that entry) or a scalar.

    Args:
        a: Matrix of shape (m, n).
        b: Vector of shape (m,) or matrix of shape (m, k).
        constraints: Vector or matrix of constraints.

    Returns:
        x: The least-squares solution.

    """
    if b.ndim == 1:
        return _lstsq_vector(a, b, constraints)

    return _lstsq_matrix(a.T, b.T, constraints).T


def lstsq_ransac(a, b, num_iter, sample_size, min_num_inliers, tol, constraints=None):
    best_x = None
    best_err = float('inf')

    for i in range(num_iter):
        # Randomly select some distinct pairs of rows (a_i, b_i)
        sample = random.sample(range(a.shape[0]), min(sample_size, a.shape[0]))

        a_ = a[sample]
        b_ = b[sample]

        # Estimate a model with these pairs
        x = lstsq(a_, b_, constraints)

        also_inliers = []

        # For every other pair
        for k in set(range(a.shape[0])).difference(sample):
            ak = a[k]
            bk = b[k]

            # If it agrees with the model keep it
            if np.linalg.norm(bk - np.matmul(ak, x)) < tol:
                also_inliers.append(k)

        # If there is enough data that agrees with this model
        if len(also_inliers) >= min_num_inliers:
            a_ = a[sample + also_inliers]
            b_ = b[sample + also_inliers]

            # Estimate a final model with all the data
            x = lstsq(a_, b_, constraints)

            err = np.linalg.norm(b_ - np.matmul(a_, x))

            # If this estimate is better than the last one
            if err < best_err:
                # Update the best model
                best_x = x
                best_err = err

    return best_x
