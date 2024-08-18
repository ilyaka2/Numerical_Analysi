
# git link:

import numpy as np
from scipy.interpolate import CubicSpline

#lagrange
def lagrange_interpolation(x_data, y_data, x):
    """
    Lagrange Interpolation

    Parameters:
    x_data (list): List of x-values for data points.
    y_data (list): List of y-values for data points.
    x (float): The x-value where you want to evaluate the interpolated polynomial.

    Returns:
    float: The interpolated y-value at the given x.
    """
    n = len(x_data)
    result = 0.0

    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term *= (x - x_data[j]) / (x_data[i] - x_data[j])
        result += term

    return result

# polynominal
def polynomial_approximation(x_data, y_data, x):
    """
    Polynomial Approximation Interpolation

    Parameters:
    x_data (list): List of x-values for data points.
    y_data (list): List of y-values for data points.
    x (float): The x-value where you want to evaluate the interpolated polynomial.

    Returns:
    float: The interpolated y-value at the given x.
    """
    # Create a Vandermonde matrix
    A = np.vander(x_data, increasing=True)
    # Solve the linear system to find the polynomial coefficients
    coeffs = np.linalg.solve(A, y_data)
    # Evaluate the polynomial at x
    return sum([coeffs[i] * (x ** i) for i in range(len(coeffs))])

#linear
def linear_interpolation(x_data, y_data, x):
    """
    Linear Interpolation

    Parameters:
    x_data (list): List of x-values for data points.
    y_data (list): List of y-values for data points.
    x (float): The x-value where you want to evaluate the interpolated value.

    Returns:
    float: The interpolated y-value at the given x.
    """
    for i in range(len(x_data) - 1):
        if x_data[i] <= x <= x_data[i + 1]:
            x1, y1 = x_data[i], y_data[i]
            x2, y2 = x_data[i + 1], y_data[i + 1]
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    raise ValueError("x is out of the interpolation range.")

if __name__ == '__main__':
    x_data = [1, 2, 5]
    y_data = [1, 0, 2]
    x_interpolate = 3  # The x-value where you want to interpolate

    # Prompt the user to choose the interpolation method
    print("Choose the interpolation method:")
    print("1. Lagrange Interpolation")
    print("2. Polynomial Approximation")
    print("3. Linear Interpolation")
    choice = input("Enter the number corresponding to your choice: ")

    if choice == '1':
        y_interpolate = lagrange_interpolation(x_data, y_data, x_interpolate)
        method_name = "Lagrange Interpolation"
    elif choice == '2':
        y_interpolate = polynomial_approximation(x_data, y_data, x_interpolate)
        method_name = "Polynomial Approximation"
    elif choice == '3':
        y_interpolate = linear_interpolation(x_data, y_data, x_interpolate)
        method_name = "Linear Interpolation"
    else:
        print("Invalid choice. Please restart the program and choose a valid option.")
        exit()

    print(f"\n{method_name}:\nInterpolated value at x = {x_interpolate} is y = {y_interpolate}")