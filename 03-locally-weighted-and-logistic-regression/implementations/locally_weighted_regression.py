import numpy as np


def gaussian_weight(x_query, x_i, tau):

    distance = np.linalg.norm(x_query - x_i)

    weight = np.exp(-(distance ** 2) / (2 * tau ** 2))

    return weight


def locally_weighted_regression(X, y, x_query, tau):

    m = X.shape[0]

    weights = np.zeros((m, m))

    for i in range(m):

        weights[i, i] = gaussian_weight(
            x_query,
            X[i],
            tau
        )

    X_transpose = X.T

    theta = np.linalg.inv(
        X_transpose @ weights @ X
    ) @ X_transpose @ weights @ y

    prediction = x_query @ theta

    return prediction, theta


if __name__ == "__main__":

    X = np.array([
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4]
    ])

    y = np.array([
        [1],
        [2],
        [2.5],
        [4]
    ])

    x_query = np.array([1, 2.5])

    tau = 0.5

    prediction, theta = locally_weighted_regression(
        X,
        y,
        x_query,
        tau
    )

    print("Prediction:")
    print(prediction)

    print("\nLocal Theta:")
    print(theta)
