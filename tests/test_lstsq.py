import itertools

import numpy as np

from cocktail import lstsq, lstsq_ransac


def test_lstsq_none_constraints_is_solution_for_vectors():
    for m, n in itertools.permutations([10, 20]):
        # Solution
        x = np.random.randint(-100, 101, size=n)

        A = np.random.randint(-100, 101, size=(m, n))
        b = np.matmul(A, x)

        # Result
        r = lstsq(A, b)

        # Test result is a solution
        assert np.allclose(np.matmul(A, r), b)


def test_lstsq_nan_constraints_is_solution_for_vectors():
    for m, n in itertools.permutations([10, 20]):
        # Solution
        x = np.random.randint(-100, 101, size=n)

        A = np.random.randint(-100, 101, size=(m, n))
        b = np.matmul(A, x)

        # Constraints
        constraints = np.empty(n)
        constraints[:] = np.nan

        # Result
        r = lstsq(A, b, constraints)

        # Test result is a solution
        assert np.allclose(np.matmul(A, r), b)


def test_lstsq_with_constraints_is_solution_for_vectors():
    for m, n in itertools.permutations([10, 20]):
        # Solution
        x = np.random.randint(-100, 101, size=n)

        A = np.random.randint(-100, 101, size=(m, n))
        b = np.matmul(A, x)

        # Constraints
        constraints = np.empty(n)
        constraints[:] = np.nan

        size = 5
        i = np.random.randint(n - size + 1)

        constraints[i:i+size] = x[i:i+size]

        # Result
        r = lstsq(A, b, constraints)

        # Test result is a solution
        assert np.allclose(np.matmul(A, r), b)
        # Test result has constrained elements
        assert np.allclose(r[i:i+size], constraints[i:i+size])


def test_lstsq_none_constraints_is_solution_for_matrices():
    for m, n, k in itertools.permutations([10, 20, 30]):
        # Solution
        A = np.random.randint(-100, 101, size=(m, n))

        X = np.random.randint(-100, 101, size=(n, k))
        Y = np.matmul(A, X)

        # Result
        r = lstsq(X.T, Y.T).T

        # Test result is a solution
        assert np.allclose(np.matmul(r, X), Y)


def test_lstsq_nan_constraints_is_solution_for_matrices():
    for m, n, k in itertools.permutations([10, 20, 30]):
        # Solution
        A = np.random.randint(-100, 101, size=(m, n))

        X = np.random.randint(-100, 101, size=(n, k))
        Y = np.matmul(A, X)

        # Constraints
        constraints = np.empty((m, n))
        constraints[:] = np.nan

        # Result
        r = lstsq(X.T, Y.T, constraints).T

        # Test result is a solution
        assert np.allclose(np.matmul(r, X), Y)


def test_lstsq_with_constraints_is_solution_for_matrices():
    for m, n, k in itertools.permutations([10, 20, 30]):
        # Solution
        A = np.random.randint(-100, 101, size=(m, n))

        X = np.random.randint(-100, 101, size=(n, k))
        Y = np.matmul(A, X)

        # Constraints
        constraints = np.empty((m, n))
        constraints[:] = np.nan

        size = 5
        i = np.random.randint(m - size + 1)
        j = np.random.randint(n - size + 1)

        constraints[i:i+size, j:j+size] = A[i:i+size, j:j+size]

        # Result
        r = lstsq(X.T, Y.T, constraints).T

        # Test result is a solution
        assert np.allclose(np.matmul(r, X), Y)
        # Test result has constrained elements
        assert np.allclose(r[i:i+size, j:j+size], constraints[i:i+size, j:j+size])


def test_lstsq_none_and_nan_constraints_coincide_for_vectors():
    for m, n in itertools.permutations([10, 20]):
        # Solution
        x = np.random.randint(-100, 101, size=n)

        A = np.random.randint(-100, 101, size=(m, n))
        b = np.matmul(A, x)

        # Constraints
        constraints = np.empty(n)
        constraints[:] = np.nan

        # Results
        results = [
            lstsq(A, b),
            lstsq(A, b, constraints)
        ]

        # Test they are all close to each other
        assert np.alltrue([np.allclose(ri, rj) for ri, rj in itertools.combinations(results, r=2)])


def test_lstsq_none_and_nan_constraints_coincide_for_matrices():
    for m, n, k in itertools.permutations([10, 20, 30]):
        # Solution
        A = np.random.randint(-100, 101, size=(m, n))

        X = np.random.randint(-100, 101, size=(n, k))
        Y = np.matmul(A, X)

        # Constraints
        constraints = np.empty((m, n))
        constraints[:] = np.nan

        # Results
        results = [
            lstsq(X.T, Y.T).T,
            lstsq(X.T, Y.T, constraints).T
        ]

        # Test they are all close to each other
        assert np.alltrue([np.allclose(ri, rj) for ri, rj in itertools.combinations(results, r=2)])


def test_lstsq_ransac_is_robust_solution():
    k = 50
    size = 10

    # k random points in the square [-size/2, size/2) x [-size/2, size/2)
    # Each column is a 2D point
    X = size * np.random.random(size=(2, k)) - size / 2

    # X plus a last row of ones
    X = np.concatenate((X, np.ones((1, X.shape[1]))))

    sx = 2
    sy = 3
    tx = -4
    ty = 1

    # Unobservable 2D affine transformation
    A = np.array([
        [sx, 0, tx],
        [0, sy, ty]
    ])

    # Transformed points
    Y = np.matmul(A, X)

    p = 0.5  # Ratio of inliers
    r = int(p * k)  # Number of inliers

    # Last k - r points are not from y = A*x model but random
    Y[:, r:] = size * np.random.random(size=(Y.shape[0], k - r)) - size / 2

    constraints = np.empty((2, 3))
    constraints[:] = np.nan
    constraints[0, 1] = constraints[1, 0] = 0

    # This yields a bad model!
    x1 = lstsq(X.T, Y.T, constraints).T
    assert not np.allclose(x1, A)

    num_iter = 100
    sample_size = 2
    min_num_inliers = 8
    tol = 0.05

    constraints = np.empty((2, 3))
    constraints[:] = np.nan
    constraints[0, 1] = constraints[1, 0] = 0

    # This yields a good model
    x2 = lstsq_ransac(X.T, Y.T, num_iter, sample_size, min_num_inliers, tol, constraints).T
    assert np.allclose(x2, A)
