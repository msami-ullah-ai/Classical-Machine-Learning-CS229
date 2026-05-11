import numpy as np


def softmax(z):

    z = z - np.max(z, axis=1, keepdims=True)

    exp_values = np.exp(z)

    probabilities = exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )

    return probabilities


X = np.array([
    [2, 1],
    [1, 3],
    [4, 2]
])

theta = np.array([
    [0.2, 0.5, 0.1],
    [0.4, 0.2, 0.3]
])

scores = X @ theta

probabilities = softmax(scores)

print("Class Scores:")
print(scores)

print("\nSoftmax Probabilities:")
print(probabilities)

print("\nPredicted Classes:")
print(np.argmax(probabilities, axis=1))
