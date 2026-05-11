# Worked Example of Gradient Descent

## Goal

In this example, we will manually perform one iteration of Gradient Descent.

This demonstrates how:

- predictions are computed,
- errors are calculated,
- parameters are updated.

---

# Training Data

Suppose we have:

| x | y |
|---|---|
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |

---

# Initial Parameters

Initialize:

$$
\theta_0=0
$$

$$
\theta_1=0
$$

Learning rate:

$$
\alpha=0.1
$$

---

# Hypothesis Function

The hypothesis is:

$$
h_\theta(x)=\theta_0+\theta_1x
$$

Initially:

$$
h_\theta(x)=0
$$

for all examples.

---

# Step 1 — Compute Predictions

For each example:

| x | Prediction |
|---|---|
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |

---

# Step 2 — Compute Errors

Error:

$$
h_\theta(x)-y
$$

| x | y | Prediction | Error |
|---|---|---|---|
| 1 | 1 | 0 | -1 |
| 2 | 2 | 0 | -2 |
| 3 | 3 | 0 | -3 |

---

# Step 3 — Compute Gradient for $\\theta_0$

Formula:

$$
\frac{\partial}{\partial\theta_0}J(\theta)
=
\frac{1}{m}
\sum_{i=1}^{m}
(h_\theta(x^{(i)})-y^{(i)})
$$

Substitute values:

$$
=
\frac{1}{3}
(-1-2-3)
$$

$$
=
\frac{-6}{3}
$$

$$
=-2
$$

---

# Step 4 — Compute Gradient for $\\theta_1$

Formula:

$$
\frac{\partial}{\partial\theta_1}J(\theta)
=
\frac{1}{m}
\sum_{i=1}^{m}
(h_\theta(x^{(i)})-y^{(i)})x^{(i)}
$$

Substitute values:

$$
=
\frac{1}{3}
[
(-1)(1)+(-2)(2)+(-3)(3)
]
$$

$$
=
\frac{1}{3}
(-1-4-9)
$$

$$
=
\frac{-14}{3}
$$

$$
\approx -4.67
$$

---

# Step 5 — Update Parameters

Update rule:

$$
\theta_j
:=
\theta_j
-
\alpha
\frac{\partial}{\partial\theta_j}J(\theta)
$$

---

## Update $\\theta_0$

$$
\theta_0
=
0
-
0.1(-2)
$$

$$
=0.2
$$

---

## Update $\\theta_1$

$$
\theta_1
=
0
-
0.1(-4.67)
$$

$$
\approx0.467
$$

---

# Updated Parameters

After one iteration:

$$
\theta_0=0.2
$$

$$
\theta_1\approx0.467
$$

---

# New Hypothesis

The updated model becomes:

$$
h_\theta(x)=0.2+0.467x
$$

This fits the data much better than the initial model.

---

# What Happened Intuitively?

Initially:

- predictions were too small,
- errors were negative.

Gradient Descent corrected this by:

- increasing parameters,
- moving predictions closer to actual values.

---

# Repeating the Process

Gradient Descent repeats:

1. prediction,
2. error computation,
3. gradient computation,
4. parameter update.

After many iterations:

- cost decreases,
- predictions improve,
- parameters converge near optimal values.

---

# Why This Example is Important

This example demonstrates the complete optimization process manually.

It connects:

- calculus,
- optimization,
- parameter updates,
- prediction improvement.

---

# Key Insights

- Errors drive parameter updates.
- Gradients determine correction direction.
- Learning rate controls update size.
- Repeated updates reduce cost gradually.

---

# Common Mistakes

- Forgetting to average over $m$
- Updating parameters incorrectly
- Confusing prediction with error
- Ignoring negative signs in gradients

---

# Implementation Perspective

In practice:

- these calculations are vectorized,
- thousands of updates happen automatically,
- libraries like NumPy compute gradients efficiently.
