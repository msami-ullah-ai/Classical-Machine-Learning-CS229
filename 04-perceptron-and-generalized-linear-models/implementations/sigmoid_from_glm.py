import numpy as np


def sigmoid(z):

    return 1 / (1 + np.exp(-z))


def linear_predictor(X, theta):

    return X @ theta


X = np.array([
    [1, 2],
    [2, 1],
    [3, 4]
])

theta = np.array([
    [0.5],
    [1.2]
])

eta = linear_predictor(X, theta)

probabilities = sigmoid(eta)

print("Linear Predictor (eta):")
print(eta)

print("\nSigmoid Probabilities:")
print(probabilities)
