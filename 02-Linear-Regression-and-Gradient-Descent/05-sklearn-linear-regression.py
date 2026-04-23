import numpy as np
from sklearn.linear_model import LinearRegression
import time

# Multivariable dataset
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
])

y = np.array([8, 13, 18, 23, 28])

# Create model
model = LinearRegression()

start = time.time()

# Train model
model.fit(X, y)

end = time.time()

# Learned parameters
w = model.coef_
b = model.intercept_

print("Weights (w):", w)
print("Bias (b):", b)

# Predictions
pred = model.predict(X)
print("Predictions:", pred)

print("Start Time:", start)
print("End Time:", end)
