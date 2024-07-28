import numpy as np
from numpy.linalg import norm

"""
Solves the system of linear equations Ax = b using iterative methods.

Jacobi and Gauss-Seidel methods are implemented to approximate the solution.

Parameters:
    a_matrix (numpy.ndarray): The NxN coefficient matrix A.
    b_vector (numpy.ndarray): The N-dimensional right-hand side vector b.
    initial_guess (numpy.ndarray): Initial guess for the solution vector x.
    tolerance (float, optional): Convergence tolerance; iteration stops when the change in x is less than this value. 
                                 Default is 1e-16.
    max_iterations (int, optional): Maximum number of iterations. Default is 200.

Returns:
    tuple: The estimated solution vector x.
"""


def jacobi_iterative(a_matrix, b_vector, initial_guess, tolerance=1e-16, max_iterations=200):
    n = len(a_matrix)
    k = 1

    if not is_diagonally_dominant(a_matrix):
        raise ValueError("The matrix is not diagonally dominant.")

    print(
        "Iteration" + "\t\t\t".join([" {:>12}".format(var)
                                     for var in ["x{}".format(i)
                                                 for i in range(1, len(a_matrix) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")

    while k <= max_iterations:
        x = np.zeros(n, dtype=np.double)
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += a_matrix[i][j] * initial_guess[j]
            x[i] = (b_vector[i] - sigma) / a_matrix[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - initial_guess, np.inf) < tolerance:
            return tuple(x)

        k += 1
        initial_guess = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)


def gauss_seidel(a_matrix, b_vector, initial_guess, tolerance=1e-16, max_iterations=200):
    n = len(a_matrix)
    k = 1

    if not is_diagonally_dominant(a_matrix):
        raise ValueError("The matrix is not diagonally dominant.")

    print(
        "Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(a_matrix) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")
    x = np.zeros(n, dtype=np.double)
    while k <= max_iterations:

        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += a_matrix[i][j] * x[j]
            x[i] = (b_vector[i] - sigma) / a_matrix[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - initial_guess, np.inf) < tolerance:
            return tuple(x)

        k += 1
        initial_guess = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)


def is_diagonally_dominant(matrix):
    for i in range(len(matrix)):
        sum_row = sum(abs(matrix[i][j]) for j in range(len(matrix)) if j != i)
        if abs(matrix[i][i]) <= sum_row:
            return False
    return True


if __name__ == "__main__":
    A = np.array([
        [3, -1, 1],
        [0, 2, -1],
        [1, 1, -3]
    ])

    user_input = False
    u_tolerance = 0.001
    u_iterations = 1000
    b = np.array([4, -1, -3])
    x = np.zeros_like(b, dtype=np.double)
    guess = np.zeros_like(b)

    try:
        print("1. Jacobi Iterative Method")
        if user_input:
            solution = jacobi_iterative(A, b, x, u_tolerance, u_iterations)
        else:
            solution = jacobi_iterative(A, b, x)
        print("\nApproximate solution:", solution)
    except ValueError as err:
        print(err)

    try:
        print("\n2. Gauss Seidel Method")
        if user_input:
            solution2 = gauss_seidel(A, b, guess, u_tolerance, u_iterations)
        else:
            solution2 = gauss_seidel(A, b, guess)
        print("\nApproximate solution:", solution2)
    except ValueError as err:
        print(err)
