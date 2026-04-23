import numpy as np


# Vectorized multivariable linear regression

# pred = X @ w + b
# error = pred - y
# dj_dw = (X.T @ error) / m
# dj_db = np.sum(error) / m


# Vectorized Single Variable linear regression

def compute_cost(x, y, w, b):
    m = x.shape[0]

    pred = w*x + b
    error = pred - y

    cost = np.sum(error**2) / (2*m)
    return cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]

    pred = w*x + b
    error = pred - y

    dj_dw = np.sum(error * x) / m
    dj_db = np.sum(error) / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):

    w = w_in
    b = b_in

    for i in range(num_iters):

        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i % 100 == 0:
            cost = compute_cost(x, y, w, b)
            print(f"Iteration {i}: Cost {cost:.4f}")

    return w, b


x = np.array([1,2,3,4,5])
y = np.array([5,8,11,14,17])

w = 0
b = 0

alpha = 0.01
iters = 1000

w_final, b_final = gradient_descent(x, y, w, b, alpha, iters)

print(w_final, b_final)
