# Gradient Descent Derivation

## Goal

We want to mathematically derive the Gradient Descent update rule for Linear Regression.

The objective is to minimize the cost function.

---

## Cost Function

We start with:

:contentReference[oaicite:5]{index=5}

---

## Hypothesis Function

For multiple features:

:contentReference[oaicite:6]{index=6}

---

## Objective

We need:

\[
\frac{\partial}{\partial \theta_j}J(\theta)
\]

This derivative tells us:

- how the cost changes,
- when parameter \(\theta_j\) changes slightly.

---

## Applying the Chain Rule

Start with:

\[
J(\theta)=\frac{1}{2m}(h_\theta(x)-y)^2
\]

Differentiate with respect to \(\theta_j\):

\[
\frac{\partial}{\partial \theta_j}J(\theta)
\]

Apply chain rule:

\[
\frac{1}{2m}\cdot 2(h_\theta(x)-y)\cdot
\frac{\partial}{\partial \theta_j}(h_\theta(x)-y)
\]

The 2 cancels the \(\frac{1}{2}\).

Result:

\[
=\frac{1}{m}(h_\theta(x)-y)
\frac{\partial}{\partial \theta_j}(h_\theta(x)-y)
\]

---

## Differentiate the Hypothesis

Since:

\[
h_\theta(x)=\sum_{j=0}^{n}\theta_jx_j
\]

Differentiating with respect to \(\theta_j\):

\[
\frac{\partial}{\partial \theta_j}h_\theta(x)=x_j
\]

Therefore:

\[
\frac{\partial}{\partial \theta_j}J(\theta)
=
\frac{1}{m}(h_\theta(x)-y)x_j
\]

---

## Final Gradient

For all training examples:

:contentReference[oaicite:7]{index=7}

---

## Final Gradient Descent Update Rule

Substitute gradient into update equation:

:contentReference[oaicite:8]{index=8}

---

## Important Intuition

A very useful interpretation:

> parameter update = prediction error × importance of feature

Where:

- prediction error tells how wrong the model is,
- feature value determines how much that feature contributed.

---

## Understanding the Error Term

\[
(h_\theta(x)-y)
\]

### Positive Error

Prediction too high.

Parameters should decrease.

---

### Negative Error

Prediction too low.

Parameters should increase.

---

## Why Feature Multiplication Matters

Each parameter corresponds to a feature.

Features with larger contribution should influence updates more strongly.

That is why the gradient multiplies by:

\[
x_j
\]

---

## Batch Gradient Descent Formula

Using all training examples:

\[
\theta_j:=\theta_j-
\alpha
\frac{1}{m}
\sum_{i=1}^{m}
(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
\]

This is called **Batch Gradient Descent**.

---

## Key Insights

- Derivatives measure sensitivity of cost.
- Chain rule allows differentiation of nested functions.
- Gradients determine update direction.
- Feature values influence parameter adjustments.

---

## Common Mistakes

- Forgetting chain rule
- Mixing feature index and example index
- Forgetting summation over training examples
- Confusing prediction error sign

---

## Implementation Perspective

In practice:

- gradients are computed vectorized,
- loops are avoided when possible,
- NumPy performs matrix operations efficiently.

Gradient computation becomes extremely fast with vectorization.
