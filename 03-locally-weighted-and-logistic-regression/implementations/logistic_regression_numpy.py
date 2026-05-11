import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None

    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    def add_bias(self, X):

        ones = np.ones((X.shape[0], 1))

        return np.concatenate((ones, X), axis=1)

    def compute_cost(self, X, y):

        m = len(y)

        predictions = self.sigmoid(X @ self.theta)

        cost = -(1 / m) * np.sum(
            y * np.log(predictions) +
            (1 - y) * np.log(1 - predictions)
        )

        return cost

    def fit(self, X, y):

        X = self.add_bias(X)

        y = y.reshape(-1, 1)

        m, n = X.shape

        self.theta = np.zeros((n, 1))

        for epoch in range(self.epochs):

            predictions = self.sigmoid(X @ self.theta)

            errors = predictions - y

            gradients = (1 / m) * (X.T @ errors)

            self.theta = self.theta - self.learning_rate * gradients

            if epoch % 100 == 0:

                cost = self.compute_cost(X, y)

                print(f"Epoch {epoch} | Cost = {cost:.4f}")

    def predict_probability(self, X):

        X = self.add_bias(X)

        return self.sigmoid(X @ self.theta)

    def predict(self, X, threshold=0.5):

        probabilities = self.predict_probability(X)

        return (probabilities >= threshold).astype(int)


if __name__ == "__main__":

    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5]
    ])

    y = np.array([
        0,
        0,
        0,
        1,
        1
    ])

    model = LogisticRegression(
        learning_rate=0.1,
        epochs=1000
    )

    model.fit(X, y)

    predictions = model.predict(X)

    print("\nPredictions:")
    print(predictions)

    print("\nParameters:")
    print(model.theta)
