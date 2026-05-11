# Vectorized Linear Regression

## Why Vectorization Matters

In Machine Learning, datasets can become extremely large.

Using loops for every computation becomes:

- slow,
- inefficient,
- computationally expensive.

Vectorization allows us to perform operations on entire vectors and matrices simultaneously.

This dramatically improves performance.

---

# Scalar Form of Linear Regression

Without vectorization, the hypothesis is written as:

$$
h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+\cdots+\theta_nx_n
$$

This form becomes inconvenient when the number of features grows.

---

# Feature Vector Representation

We represent features as a vector:

$$
x=
\begin{bmatrix}
1 \\
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
$$

The extra 1 represents the bias feature.

---

# Parameter Vector

Similarly, parameters are represented as:

$$
\theta=
\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\theta_2 \\
\vdots \\
\theta_n
\end{bmatrix}
$$

---

# Vectorized Hypothesis

The hypothesis can now be written compactly as:

$$
h_\theta(x)=\theta^Tx
$$

This is simply the dot product between:

- parameter vector,
- feature vector.

---

# Expanding the Dot Product

Expanding:

$$
\theta^Tx
$$

gives:

$$
\theta_0+\theta_1x_1+\theta_2x_2+\cdots+\theta_nx_n
$$

which is exactly the original hypothesis equation.

---

# Matrix Representation of the Dataset

Suppose we have:

- $m$ training examples,
- $n$ features.

We represent the dataset using matrix:

$$
X=
\begin{bmatrix}
---(x^{(1)})^T--- \\
---(x^{(2)})^T--- \\
\vdots \\
---(x^{(m)})^T---
\end{bmatrix}
$$

Dimensions:

$$
X \in \mathbb{R}^{m \times (n+1)}
$$

---

# Target Vector

Outputs are stored in:

$$
y=
\begin{bmatrix}
y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(m)}
\end{bmatrix}
$$

Dimensions:

$$
y \in \mathbb{R}^{m \times 1}
$$

---

# Parameter Dimensions

Parameter vector dimensions:

$$
\theta \in \mathbb{R}^{(n+1)\times1}
$$

---

# Vectorized Predictions

Predictions for all training examples are computed simultaneously:

$$
X\theta
$$

Dimensions:

$$
(m \times (n+1))
\times
((n+1)\times1)
=
(m\times1)
$$

Result:

$$
X\theta=
\begin{bmatrix}
h_\theta(x^{(1)}) \\
h_\theta(x^{(2)}) \\
\vdots \\
h_\theta(x^{(m)})
\end{bmatrix}
$$

---

# Vectorized Cost Function

The cost function becomes:

$$
J(\theta)=\frac{1}{2m}(X\theta-y)^T(X\theta-y)
$$

Where:

$$
(X\theta-y)
$$

represents the error vector.

---

# Understanding the Error Vector

The error vector contains prediction errors for all examples:

$$
X\theta-y=
\begin{bmatrix}
h_\theta(x^{(1)})-y^{(1)} \\
h_\theta(x^{(2)})-y^{(2)} \\
\vdots \\
h_\theta(x^{(m)})-y^{(m)}
\end{bmatrix}
$$

---

# Why Transpose Appears

The transpose:

$$
(X\theta-y)^T
$$

converts the column vector into a row vector.

This allows matrix multiplication:

$$
(1\times m)(m\times1)
$$

resulting in a scalar value.

---

# Vectorized Gradient Descent

The vectorized gradient descent update rule is:

$$
\theta :=
\theta
-
\alpha
\frac{1}{m}
X^T(X\theta-y)
$$

This updates all parameters simultaneously.

---

# Why Vectorization is Powerful

Vectorization:

- removes explicit loops,
- speeds up computation,
- enables GPU acceleration,
- simplifies implementation.

Modern ML frameworks rely heavily on vectorized operations.

---

# Computational Efficiency

Without vectorization:

- operations are performed one example at a time.

With vectorization:

- entire datasets are processed simultaneously.

This becomes critical for large-scale machine learning.

---

# Dimensions Summary

| Quantity | Dimensions |
|---|---|
| $X$ | $m\times(n+1)$ |
| $\theta$ | $(n+1)\times1$ |
| $y$ | $m\times1$ |
| $X\theta$ | $m\times1$ |

---

# Common Dimension Mistakes

A very common beginner mistake is mismatched matrix dimensions.

Always verify:

$$
(A_{m\times n})(B_{n\times p})
$$

Inner dimensions must match.

---

# Key Insights

- Vectorization replaces loops with matrix operations.
- The hypothesis becomes a dot product.
- Predictions for all examples are computed simultaneously.
- Vectorization dramatically improves efficiency.

---

# Common Mistakes

- Forgetting the bias column
- Confusing matrix dimensions
- Misusing transpose operations
- Mixing row vectors and column vectors

---

# Implementation Perspective

In NumPy:

```python
predictions = X @ theta
errors = predictions - y
cost = (1/(2*m)) * (errors.T @ errors)
```

Modern ML libraries are optimized for these vectorized computations.
