import numpy as np
import time

# Iterative optimization method:

# θ := θ − α ∇J(θ)

# ∇J(θ) = 1/m*(​X^T(Xθ−y))

# θ := θ − α*(1/m*​X^T(Xθ−y))

# w = w - alpha * dj_dw
# b = b - alpha * dj_db


def compute_gradient(x, y, w, b):
    dj_dw = 0
    dj_db = 0
    m = x.shape[0]
    for i in range(m):
        f_wb = w*x[i] + b
        dj_dw_i = (f_wb-y[i])*x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = (1/m)*dj_dw
    dj_db = (1/m)*dj_db
    return dj_dw, dj_db


def cost_function(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w*x[i] + b
        error = f_wb - y[i]
        cost += error**2
    
    return 1/(2*m) * cost
 

def gradient_descent(x, y, w_in, b_in, learning_rate, num_iters):
    
    w = w_in
    b = b_in
    
    for i in range(num_iters):
        
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        
        if i % 100 == 0:
            cost = cost_function(x, y, w, b)
            print(f"Iteration {i}: Cost {cost:.4f}")
    
    return w, b


# EXAMPLE

x = np.array([1,2,3,4,5])
y = np.array([5,8,11,14,17])

w = 0
b = 0

alpha = 0.01
iterations = 1000

start = time.time()
w_final, b_final = gradient_descent(x, y, w, b, alpha, iterations)
end = time.time()

print(w_final, b_final)

print("Start Time:", start)
print("End Time:", end)
