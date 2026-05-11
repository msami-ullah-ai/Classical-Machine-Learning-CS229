import numpy as np


def sigmoid(z):

    return 1 / (1 + np.exp(-z))


def newtons_method(X, y, epochs=10):

    m, n = X.shape

    theta = np.zeros((n, 1))

    for epoch in range(epochs):

        predictions = sigmoid(X @ theta)

        gradient = (1 / m) * (X.T @ (predictions - y))

        diagonal = predictions * (1 - predictions)

        R = np.diag(diagonal.flatten())

        hessian = (1 / m) * (X.T @ R @ X)

        theta = theta - np.linalg.inv(hessian) @ gradient

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
        [0],
        [0],
        [1],
        [1]
    ])

    theta = newtons_method(X, y)

    print("\nOptimized Parameters:")
    print(theta)
