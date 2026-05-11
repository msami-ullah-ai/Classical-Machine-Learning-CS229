"""
Linear Regression From Scratch using NumPy

This implementation follows the mathematical derivations
covered in the lecture notes.

Concepts implemented:
- Hypothesis Function
- Cost Function
- Gradient Descent
- Vectorized Operations

Author: Sami Ullah
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression using Batch Gradient Descent.
    """

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None
        self.cost_history = []

    def add_bias(self, X):
        """
        Adds bias feature x0 = 1 to the dataset.
        """

        ones = np.ones((X.shape[0], 1))
        return np.concatenate((ones, X), axis=1)

    def compute_cost(self, X, y):
        """
        Computes Mean Squared Error Cost Function.

        J(theta) = (1/2m) * sum((predictions - y)^2)
        """

        m = len(y)

        predictions = X @ self.theta
        errors = predictions - y

        cost = (1 / (2 * m)) * np.sum(errors ** 2)

        return cost

    def fit(self, X, y):
        """
        Trains Linear Regression using Gradient Descent.
        """

        X = self.add_bias(X)

        m, n = X.shape

        # Initialize parameters with zeros
        self.theta = np.zeros((n, 1))

        y = y.reshape(-1, 1)

        for epoch in range(self.epochs):

            # Vectorized predictions
            predictions = X @ self.theta

            # Compute errors
            errors = predictions - y

            # Compute gradients
            gradients = (1 / m) * (X.T @ errors)

            # Update parameters
            self.theta = self.theta - self.learning_rate * gradients

            # Store cost for analysis
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Cost = {cost:.4f}")

    def predict(self, X):
        """
        Predicts outputs for new data.
        """

        X = self.add_bias(X)

        return X @ self.theta


if __name__ == "__main__":

    # Example dataset
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5]
    ])

    y = np.array([2, 4, 6, 8, 10])

    model = LinearRegression(
        learning_rate=0.01,
        epochs=1000
    )

    model.fit(X, y)

    predictions = model.predict(X)

    print("\nFinal Parameters:")
    print(model.theta)

    print("\nPredictions:")
    print(predictions)
