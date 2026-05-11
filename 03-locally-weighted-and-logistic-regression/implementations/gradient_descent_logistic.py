import numpy as np


def sigmoid(z):

    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, theta):

    m = len(y)

    predictions = sigmoid(X @ theta)

    cost = -(1 / m) * np.sum(
        y * np.log(predictions) +
        (1 - y) * np.log(1 - predictions)
    )

    return cost


def gradient_descent(X, y, theta, learning_rate, epochs):

    m = len(y)

    cost_history = []

    for epoch in range(epochs):

        predictions = sigmoid(X @ theta)

        errors = predictions - y

        gradients = (1 / m) * (X.T @ errors)

        theta = theta - learning_rate * gradients

        cost = compute_cost(X, y, theta)

        cost_history.append(cost)

        if epoch % 100 == 0:

            print(f"Epoch {epoch} | Cost = {cost:.4f}")

    return theta, cost_history


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

    theta = np.zeros((2, 1))

    theta, history = gradient_descent(
        X,
        y,
        theta,
        learning_rate=0.1,
        epochs=1000
    )

    print("\nFinal Parameters:")
    print(theta)
