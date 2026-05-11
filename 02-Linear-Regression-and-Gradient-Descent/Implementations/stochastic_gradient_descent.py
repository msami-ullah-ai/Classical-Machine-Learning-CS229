"""
Stochastic Gradient Descent From Scratch

Unlike Batch Gradient Descent,
SGD updates parameters using ONE example at a time.

Author: Sami Ullah
"""

import numpy as np


def stochastic_gradient_descent(
    X,
    y,
    theta,
    learning_rate=0.01,
    epochs=50
):

    m = len(y)

    for epoch in range(epochs):

        for i in range(m):

            # Select single training example
            xi = X[i:i+1]
            yi = y[i:i+1]

            # Prediction
            prediction = xi @ theta

            # Error
            error = prediction - yi

            # Gradient
            gradient = xi.T @ error

            # Parameter update
            theta = theta - learning_rate * gradient

        print(f"Epoch {epoch + 1} completed")

    return theta


if __name__ == "__main__":

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

    theta = np.zeros((2, 1))

    theta = stochastic_gradient_descent(
        X,
        y,
        theta,
        learning_rate=0.01,
        epochs=20
    )

    print("\nLearned Parameters:")
    print(theta)
