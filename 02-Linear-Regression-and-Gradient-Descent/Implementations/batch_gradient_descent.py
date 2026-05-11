"""
Batch Gradient Descent Implementation

This script demonstrates how Batch Gradient Descent
updates parameters using the ENTIRE dataset.

"""

import numpy as np


def compute_cost(X, y, theta):
    """
    Computes Mean Squared Error Cost.
    """

    m = len(y)

    predictions = X @ theta

    errors = predictions - y

    cost = (1 / (2 * m)) * np.sum(errors ** 2)

    return cost


def batch_gradient_descent(X, y, theta, learning_rate, epochs):

    """
    Performs Batch Gradient Descent.
    """

    m = len(y)

    cost_history = []

    for epoch in range(epochs):

        # Step 1: Predictions
        predictions = X @ theta

        # Step 2: Errors
        errors = predictions - y

        # Step 3: Compute gradients
        gradients = (1 / m) * (X.T @ errors)

        # Step 4: Update parameters
        theta = theta - learning_rate * gradients

        # Store cost
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Cost = {cost:.4f}")

    return theta, cost_history


if __name__ == "__main__":

    # Example dataset

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

    theta, history = batch_gradient_descent(
        X,
        y,
        theta,
        learning_rate=0.01,
        epochs=1000
    )

    print("\nOptimized Parameters:")
    print(theta)
