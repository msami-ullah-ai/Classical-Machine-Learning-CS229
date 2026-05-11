import numpy as np


def sigmoid(z):

    return 1 / (1 + np.exp(-z))


values = np.array([
    -10,
    -5,
    -1,
    0,
    1,
    5,
    10
])

results = sigmoid(values)

for value, result in zip(values, results):

    print(f"sigmoid({value}) = {result:.4f}")
