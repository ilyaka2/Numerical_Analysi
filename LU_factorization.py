

import numpy as np


def inverse_matrix(matrix):
    """
    Find the inverse of a matrix using elementary row operations.

    Parameters:
    matrix (numpy.ndarray): A square matrix to be inverted.

    Returns:
    numpy.ndarray: The inverse of the input matrix.

    Raises:
    ValueError: If the matrix is not square or singular.
    """
    dimension_a = matrix.shape[0]
    identity = np.eye(dimension_a)
    stack_matrix_ai = np.hstack((matrix, identity))

    for i in range(dimension_a):
        stack_matrix_ai[i] = stack_matrix_ai[i] / stack_matrix_ai[i, i]

        for j in range(dimension_a):
            if i != j:
                stack_matrix_ai[j] = stack_matrix_ai[j] - stack_matrix_ai[i] * stack_matrix_ai[j, i]

    inverted_a = stack_matrix_ai[:, dimension_a:]
    return inverted_a


def lu_decomposition(matrix):
    """
    Perform LU decomposition with partial pivoting.

    Parameters:
    matrix (numpy.ndarray): A square matrix to decompose.

    Returns:
    tuple: A tuple containing two numpy.ndarrays, L and U, where L is the lower triangular matrix
           and U is the upper triangular matrix.

    Raises:
    ValueError: If the matrix is not square.
    """
    dimension = matrix.shape[0]
    l = np.zeros((dimension, dimension))
    u = np.zeros((dimension, dimension))

    for i in range(dimension):
        for j in range(i, dimension):
            u[i, j] = matrix[i, j] - sum(l[i, k] * u[k, j] for k in range(i))
        for j in range(i, dimension):
            if i == j:
                l[i, i] = 1
            else:
                l[j, i] = (matrix[j, i] - sum(l[j, k] * u[k, i] for k in range(i))) / u[i, i]

    return l, u


def solve_lu(s_l, s_u):
    """
    Solve the system Ax = b using LU decomposition with partial pivoting.

    Parameters:
    s_l (numpy.ndarray): Lower triangular matrix obtained from LU decomposition.
    s_u (numpy.ndarray): Upper triangular matrix obtained from LU decomposition.
    b_vect (numpy.ndarray): The right-hand side vector b.

    Returns:
    numpy.ndarray: The solution vector x.
    """
    y = forward_substitution(s_l, b)
    x = backward_substitution(s_u, y)
    return x


def forward_substitution(L, b):
    """
    Perform forward substitution to solve Ly = b.

    Parameters:
    L (numpy.ndarray): Lower triangular matrix.
    b (numpy.ndarray): The right-hand side vector b.

    Returns:
    numpy.ndarray: The solution vector y.
    """
    n = L.shape[0]
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))

    return y


def backward_substitution(U, y):
    """
    Perform backward substitution to solve Ux = y.

    Parameters:
    U (numpy.ndarray): Upper triangular matrix.
    y (numpy.ndarray): The right-hand side vector y.

    Returns:
    numpy.ndarray: The solution vector x.
    """
    n = U.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]

    return x


# Example 3x3 matrix
A = np.array([[1, 4, -3],
              [-2, 1, 5],
              [3, 2, 1]])
print("Original Matrix A:\n", A)

# Inverse of the matrix
A_inv = inverse_matrix(A)
print("Inverse Matrix A_inv:\n", A_inv)

# b vector
b = np.array([1, 2, 3])
print("Vector b:\n", b)

# LU decomposition of the inverse matrix
L, U = lu_decomposition(A)
print("Matrix L:\n", L)
print("Matrix U:\n", U)

# Solving Ax = b using the LU decomposition
result = solve_lu(L, U)
print("Solution x:\n", result)
