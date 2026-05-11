import numpy as np


class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=100):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def step_function(self, z):

        if z >= 0:
            return 1

        return 0

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)

        for epoch in range(self.epochs):

            total_errors = 0

            for i in range(n_samples):

                linear_output = np.dot(X[i], self.weights) + self.bias

                prediction = self.step_function(linear_output)

                update = self.learning_rate * (y[i] - prediction)

                self.weights = self.weights + update * X[i]

                self.bias = self.bias + update

                if update != 0:
                    total_errors += 1

            print(f"Epoch {epoch + 1} | Misclassified = {total_errors}")

    def predict(self, X):

        predictions = []

        for x in X:

            linear_output = np.dot(x, self.weights) + self.bias

            prediction = self.step_function(linear_output)

            predictions.append(prediction)

        return np.array(predictions)


if __name__ == "__main__":

    X = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [6, 5],
        [7, 8],
        [8, 8]
    ])

    y = np.array([
        0,
        0,
        0,
        1,
        1,
        1
    ])

    model = Perceptron(
        learning_rate=0.1,
        epochs=10
    )

    model.fit(X, y)

    predictions = model.predict(X)

    print("\nPredictions:")
    print(predictions)

    print("\nWeights:")
    print(model.weights)

    print("\nBias:")
    print(model.bias)
