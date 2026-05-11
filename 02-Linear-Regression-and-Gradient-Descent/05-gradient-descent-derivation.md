# Gradient Descent Derivation

## Goal of the Derivation

We want to mathematically derive the Gradient Descent update rule used in Linear Regression.

The purpose is to understand:

- how parameter updates are computed,
- why derivatives appear in the update rule,
- how Gradient Descent reduces prediction error.

---

## Cost Function

We start with the Linear Regression cost function:

$$
J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2
$$

Where:

- $J(\theta)$ = cost function
- $m$ = number of training examples
- $h_\theta(x^{(i)})$ = prediction
- $y^{(i)}$ = actual value

---

## Objective

Our goal is:

$$
\min_\theta J(\theta)
$$

which means:

> find parameter values that minimize the cost function.

To do this, we compute derivatives with respect to each parameter.

---

## Why Derivatives?

The derivative tells us:

- how much the cost changes,
- when a parameter changes slightly.

If changing a parameter increases cost:
- move opposite that direction.

If changing a parameter decreases cost:
- move toward that direction.

This is the core idea behind Gradient Descent.

---

## Gradient Descent Update Rule

The update rule is:

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta)
$$

Where:

- $\theta_j$ = parameter being updated
- $\alpha$ = learning rate
- derivative = slope of cost function

---

## Understanding the Update Rule

The update rule has three important parts.

### Current Parameter

$$
\theta_j
$$

This is the current value of the parameter.

---

### Learning Rate

$$
\alpha
$$

Controls step size.

- small $\alpha$ = slow learning
- large $\alpha$ = aggressive updates

---

### Derivative

$$
\frac{\partial}{\partial \theta_j}J(\theta)
$$

Tells:

- direction of steepest increase,
- magnitude of change.

To minimize cost:
- move opposite the gradient.

---

# Deriving the Gradient

---

## Step 1 — Start with Cost Function

For simplicity, consider one training example first:

$$
J(\theta)=\frac{1}{2}(h_\theta(x)-y)^2
$$

---

## Step 2 — Differentiate with Respect to $\\theta_j$

We compute:

$$
\frac{\partial}{\partial \theta_j}J(\theta)
$$

Substitute the cost function:

$$
=
\frac{\partial}{\partial \theta_j}
\left[
\frac{1}{2}(h_\theta(x)-y)^2
\right]
$$

---

## Step 3 — Apply Chain Rule

Using the chain rule:

$$
=
\frac{1}{2}
\cdot
2(h_\theta(x)-y)
\cdot
\frac{\partial}{\partial \theta_j}(h_\theta(x)-y)
$$

The 2 cancels with $\frac{1}{2}$:

$$
=
(h_\theta(x)-y)
\cdot
\frac{\partial}{\partial \theta_j}(h_\theta(x)-y)
$$

---

## Step 4 — Differentiate the Hypothesis

Recall the hypothesis:

$$
h_\theta(x)=\sum_{j=0}^{n}\theta_jx_j
$$

Differentiating with respect to $\theta_j$:

$$
\frac{\partial}{\partial \theta_j}h_\theta(x)=x_j
$$

Therefore:

$$
\frac{\partial}{\partial \theta_j}(h_\theta(x)-y)=x_j
$$

because $y$ is constant.

---

## Step 5 — Final Derivative for One Example

Substitute into the equation:

$$
\frac{\partial}{\partial \theta_j}J(\theta)
=
(h_\theta(x)-y)x_j
$$

This is the gradient contribution from one training example.

---

# Extending to All Training Examples

For the full dataset:

$$
J(\theta)=
\frac{1}{2m}
\sum_{i=1}^{m}
(h_\theta(x^{(i)})-y^{(i)})^2
$$

Differentiating gives:

$$
\frac{\partial}{\partial \theta_j}J(\theta)
=
\frac{1}{m}
\sum_{i=1}^{m}
(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
$$

---

# Final Gradient Descent Update Rule

Substitute gradient into the update equation:

$$
\theta_j
:=
\theta_j
-
\alpha
\frac{1}{m}
\sum_{i=1}^{m}
(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
$$

This is the final Batch Gradient Descent formula for Linear Regression.

---

# Intuition Behind the Gradient

A powerful interpretation is:

> parameter update = prediction error × importance of feature

Where:

- prediction error tells how wrong the model is,
- feature value tells how much that feature contributed.

---

## Understanding the Error Term

The error term is:

$$
h_\theta(x^{(i)})-y^{(i)}
$$

### Positive Error

If:

$$
h_\theta(x^{(i)}) > y^{(i)}
$$

then prediction is too high.

Parameters should decrease.

---

### Negative Error

If:

$$
h_\theta(x^{(i)}) < y^{(i)}
$$

then prediction is too low.

Parameters should increase.

---

## Why Multiply by the Feature?

Each parameter corresponds to a feature.

Features contributing more strongly to the prediction should receive larger updates.

That is why the gradient contains:

$$
x_j^{(i)}
$$

---

## Why Gradient Descent Works

Gradient Descent repeatedly:

1. computes prediction errors,
2. calculates gradients,
3. updates parameters,
4. reduces cost gradually.

Over many iterations:

- predictions improve,
- cost decreases,
- parameters approach optimal values.

---

## Batch Gradient Descent

The formula above uses the entire training dataset before updating parameters.

This is called:

> Batch Gradient Descent

Characteristics:

- stable updates,
- smooth convergence,
- computationally expensive for huge datasets.

---

## Connection to Calculus

Gradient Descent is fundamentally based on calculus.

Key concepts used:

- derivatives,
- partial derivatives,
- chain rule,
- optimization.

Without calculus, modern machine learning optimization would not exist.

---

## Key Insights

- Derivatives measure sensitivity of cost.
- Chain rule helps differentiate nested functions.
- Gradients determine update direction.
- Error determines how much correction is needed.
- Features determine which parameters should change more.

---

## Common Mistakes

- Forgetting the chain rule
- Mixing feature index and example index
- Forgetting summation over training examples
- Confusing prediction error signs
- Ignoring why subtraction appears in updates

---

## Implementation Perspective

In practice:

- gradients are computed vectorized,
- loops are minimized,
- matrix operations are heavily optimized.

Libraries like NumPy compute these updates extremely efficiently for large datasets.
