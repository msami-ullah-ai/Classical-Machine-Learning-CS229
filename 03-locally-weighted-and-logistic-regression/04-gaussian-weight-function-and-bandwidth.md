# Gaussian Weight Function and Bandwidth Parameter

## Why We Need Weights in LWLR

Locally Weighted Linear Regression assumes:

> nearby points are more important than distant points.

To implement this idea mathematically, LWLR assigns weights to training examples.

Nearby examples:
- receive large weights.

Faraway examples:
- receive small weights.

---

# Gaussian Weight Function

The most common weighting function is the Gaussian kernel:

$$
w^{(i)}
=
\exp
\left(
-
\frac{(x^{(i)}-x)^2}
{2\tau^2}
\right)
$$

Where:

- $x^{(i)}$ = training example
- $x$ = query point
- $\tau$ = bandwidth parameter

---

# Understanding the Gaussian Function

The Gaussian function creates a bell-shaped curve.

Key behavior:

- nearby points receive weights close to 1,
- distant points receive weights close to 0.

---

# Intuition Behind the Formula

Suppose:

$$
(x^{(i)}-x)^2
$$

is small.

That means:

- training point is close to query point.

Result:

$$
w^{(i)} \approx 1
$$

So the example strongly influences prediction.

---

# Faraway Points

Suppose:

$$
(x^{(i)}-x)^2
$$

is large.

Then:

$$
w^{(i)} \approx 0
$$

The example contributes very little.

---

# Bell Curve Interpretation

The Gaussian function produces a smooth bell-shaped weighting curve.

Center:
- highest importance.

Far from center:
- importance decreases rapidly.

---

# Bandwidth Parameter ($\\tau$)

The parameter:

$$
\tau
$$

controls the width of the Gaussian curve.

This is one of the most important hyperparameters in LWLR.

---

# Small Bandwidth

## Behavior

Small $\tau$ creates a narrow bell curve.

Only very nearby points receive high weights.

---

## Advantages

- highly flexible,
- captures local details,
- adapts strongly to nearby patterns.

---

## Disadvantages

- sensitive to noise,
- high variance,
- may overfit.

---

# Large Bandwidth

## Behavior

Large $\tau$ creates a wider bell curve.

More points influence prediction.

---

## Advantages

- smoother predictions,
- lower variance,
- more stable behavior.

---

## Disadvantages

- less flexible,
- may ignore local structure,
- may underfit.

---

# Bias-Variance Tradeoff

Bandwidth directly controls the bias-variance tradeoff.

---

## Small $\\tau$

$$
\text{Low Bias}
$$

$$
\text{High Variance}
$$

The model fits local details closely.

---

## Large $\\tau$

$$
\text{High Bias}
$$

$$
\text{Low Variance}
$$

The model becomes smoother and less flexible.

---

# Why Bandwidth Matters

Choosing $\tau$ incorrectly can severely hurt performance.

Too small:
- overfitting.

Too large:
- underfitting.

---

# Intuition Using Neighborhoods

Imagine predicting house prices.

---

## Small $\\tau$

Only extremely nearby houses matter.

---

## Large $\\tau$

Even distant neighborhoods influence prediction.

---

# Weight Matrix Representation

Weights are often stored in diagonal matrix form:

$$
W=
\begin{bmatrix}
w^{(1)} & 0 & \cdots & 0 \\
0 & w^{(2)} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & w^{(m)}
\end{bmatrix}
$$

This matrix is used during weighted least squares optimization.

---

# Why Gaussian Kernels Are Popular

Gaussian kernels are widely used because they:

- are smooth,
- decrease gradually,
- are mathematically convenient,
- work well in practice.

---

# Relationship to Other ML Algorithms

Gaussian weighting ideas appear in:

- Support Vector Machines,
- Radial Basis Function Networks,
- Gaussian Processes,
- Kernel Methods.

---

# Key Insights

- LWLR assigns importance using weights.
- Nearby points receive larger weights.
- Gaussian kernels create smooth local weighting.
- Bandwidth controls flexibility.
- Small bandwidth increases variance.
- Large bandwidth increases bias.

---

# Common Mistakes

- Choosing bandwidth arbitrarily
- Ignoring bias-variance tradeoff
- Thinking all points contribute equally
- Forgetting weights depend on query point

---

# Implementation Perspective

In practice:

- distances are computed vectorized,
- Gaussian weights are calculated efficiently,
- weighted least squares uses these weights during optimization.

Bandwidth tuning is critical for good LWLR performance.
