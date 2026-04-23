import numpy as np

# X shape = (m, n)
# m = training examples
# n = features
# w shape = (n,)
# y shape = (m,)


def compute_gradient(X, y, w, b):
    m, n = X.shape

    dj_dw = np.zeros(n) #it would be a matrix now
    dj_db = 0.0

    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        error = f_wb - y[i]

        dj_dw += error * X[i]
        dj_db += error

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def cost_function(X, y, w, b):
    m = X.shape[0]
    cost = 0.0

    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        error = f_wb - y[i]
        cost += error**2

    return cost / (2 * m)


def gradient_descent(X, y, w_in, b_in, learning_rate, num_iters):
    w = w_in.copy()
    b = b_in

    for i in range(num_iters):

        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        if i % 100 == 0:
            cost = cost_function(X, y, w, b)
            print(f"Iteration {i}: Cost {cost:.4f}")

    return w, b
