"""
Normal Equation Implementation

Computes optimal parameters analytically
without iterative optimization.

Formula:

theta = (X^T X)^(-1) X^T y

Author: Sami Ullah
"""

import numpy as np


def normal_equation(X, y):

    """
    Computes optimal parameters
    using the Normal Equation.
    """

    X_transpose = X.T

    theta = np.linalg.inv(
        X_transpose @ X
    ) @ X_transpose @ y

    return theta


if __name__ == "__main__":

    # Dataset with bias feature included

    X = np.array([
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4]
    ])

    y = np.array([
        [2],
        [4],
        [6],
        [8]
    ])

    theta = normal_equation(X, y)

    print("Optimal Parameters:")
    print(theta)
