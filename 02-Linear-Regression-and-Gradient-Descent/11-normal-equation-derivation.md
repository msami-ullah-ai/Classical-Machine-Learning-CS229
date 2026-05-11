# Normal Equation Derivation

## Goal

We want to derive the Normal Equation mathematically.

The objective is to find:

$$
\theta
$$

that minimizes the Linear Regression cost function.

---

# Vectorized Cost Function

The vectorized cost function is:

$$
J(\theta)=\frac{1}{2m}(X\theta-y)^T(X\theta-y)
$$

Where:

- $X$ = feature matrix
- $\theta$ = parameter vector
- $y$ = target vector

---

# Expanding the Expression

Expand:

$$
(X\theta-y)^T(X\theta-y)
$$

Using transpose properties:

$$
(X\theta-y)^T
=
\theta^TX^T-y^T
$$

Therefore:

$$
J(\theta)
=
\frac{1}{2m}
(\theta^TX^T-y^T)
(X\theta-y)
$$

---

# Multiply the Terms

Expanding:

$$
=
\frac{1}{2m}
(
\theta^TX^TX\theta
-
\theta^TX^Ty
-
y^TX\theta
+
y^Ty
)
$$

---

# Important Scalar Observation

Notice:

$$
\theta^TX^Ty
$$

and:

$$
y^TX\theta
$$

are both scalars.

A scalar equals its transpose.

Therefore:

$$
\theta^TX^Ty
=
y^TX\theta
$$

This simplifies the equation.

---

# Simplified Cost Function

Now:

$$
J(\theta)
=
\frac{1}{2m}
(
\theta^TX^TX\theta
-
2\theta^TX^Ty
+
y^Ty
)
$$

---

# Differentiate with Respect to $\\theta$

Take derivative:

$$
\frac{\partial}{\partial\theta}J(\theta)
$$

Using matrix calculus rules:

$$
\frac{\partial}{\partial\theta}
(\theta^TX^TX\theta)
=
2X^TX\theta
$$

and:

$$
\frac{\partial}{\partial\theta}
(\theta^TX^Ty)
=
X^Ty
$$

---

# Final Gradient

Substituting:

$$
\frac{\partial}{\partial\theta}J(\theta)
=
\frac{1}{2m}
(
2X^TX\theta
-
2X^Ty
)
$$

Simplify:

$$
=
\frac{1}{m}
(X^TX\theta-X^Ty)
$$

---

# Setting Gradient Equal to Zero

At minimum:

$$
\frac{\partial}{\partial\theta}J(\theta)=0
$$

Therefore:

$$
X^TX\theta-X^Ty=0
$$

Move terms:

$$
X^TX\theta=X^Ty
$$

---

# Solving for $\\theta$

Multiply both sides by:

$$
(X^TX)^{-1}
$$

Result:

$$
\theta=(X^TX)^{-1}X^Ty
$$

This is the Normal Equation.

---

# Why This is Important

This derivation connects:

- calculus,
- optimization,
- matrix algebra,
- machine learning.

It shows how Linear Regression can be solved analytically.

---

# Understanding the Mathematics

The derivation relies heavily on:

- transpose properties,
- matrix multiplication,
- scalar symmetry,
- matrix differentiation.

These concepts are fundamental in advanced ML.

---

# Key Insights

- The Normal Equation comes from minimizing the cost function.
- Matrix calculus simplifies optimization.
- Setting gradient equal to zero finds the optimum.
- Linear algebra provides direct solutions to optimization problems.

---

# Common Mistakes

- Mixing transpose and inverse
- Forgetting scalar symmetry
- Using incorrect matrix dimensions
- Ignoring matrix calculus rules

---

# Implementation Perspective

Modern ML libraries rarely derive this manually.

However, understanding the derivation gives deep insight into:

- optimization,
- vectorization,
- ML mathematics,
- numerical computation.
