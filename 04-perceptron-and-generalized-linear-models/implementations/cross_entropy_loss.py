import numpy as np


def cross_entropy_loss(y_true, y_pred):

    epsilon = 1e-10

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)

    return loss


y_true = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

y_pred = np.array([
    [0.9, 0.05, 0.05],
    [0.1, 0.8, 0.1],
    [0.2, 0.2, 0.6]
])

loss = cross_entropy_loss(y_true, y_pred)

print("Cross Entropy Loss:")
print(loss)
