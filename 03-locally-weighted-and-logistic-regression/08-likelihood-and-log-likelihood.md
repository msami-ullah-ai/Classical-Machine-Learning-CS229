# Likelihood and Log-Likelihood

## Why We Need Likelihood

Logistic Regression predicts probabilities.

But we still need a way to determine:

> which parameter values best explain the observed data.

This leads to the concept of:

> Likelihood.

---

# What is Likelihood?

Likelihood measures:

> how probable the observed training data is under parameter values $\theta$.

Good parameters:

- assign high probability to correct labels.

Bad parameters:

- assign low probability to observed labels.

---

# Probability of a Single Training Example

From Logistic Regression:

$$
P(y|x;\theta)
=
h_\theta(x)^y
(1-h_\theta(x))^{1-y}
$$

Where:

$$
h_\theta(x)=g(\theta^Tx)
$$

---

# Training Dataset

Suppose we have:

$$
m
$$

training examples:

$$
(x^{(1)},y^{(1)}),\ldots,(x^{(m)},y^{(m)})
$$

---

# Independence Assumption

We assume training examples are independent.

Therefore:

- probability of observing all examples
- becomes the product of individual probabilities.

---

# Likelihood Function

The likelihood function becomes:

$$
L(\theta)
=
\prod_{i=1}^{m}
P(y^{(i)}|x^{(i)};\theta)
$$

Substitute Logistic Regression probability:

$$
L(\theta)
=
\prod_{i=1}^{m}
h_\theta(x^{(i)})^{y^{(i)}}
(1-h_\theta(x^{(i)}))^{1-y^{(i)}}
$$

---

# Goal of Maximum Likelihood Estimation

We choose parameters:

$$
\theta
$$

that maximize:

$$
L(\theta)
$$

This is called:

> Maximum Likelihood Estimation (MLE)

---

# Problem with Products

The likelihood contains a huge product:

$$
\prod
$$

Products become difficult to optimize numerically.

Especially for large datasets:

- values become extremely small,
- numerical instability occurs.

---

# Solution: Take the Log

Instead of maximizing likelihood directly:

we maximize:

$$
\ell(\theta)=\log L(\theta)
$$

called the:

> log-likelihood.

---

# Why Logarithms Help

Logarithms transform products into sums.

Using:

$$
\log(ab)=\log a+\log b
$$

we simplify optimization significantly.

---

# Deriving the Log-Likelihood

Start with:

$$
\ell(\theta)
=
\log
\left(
\prod_{i=1}^{m}
h_\theta(x^{(i)})^{y^{(i)}}
(1-h_\theta(x^{(i)}))^{1-y^{(i)}}
\right)
$$

Move log inside:

$$
=
\sum_{i=1}^{m}
\log
\left(
h_\theta(x^{(i)})^{y^{(i)}}
(1-h_\theta(x^{(i)}))^{1-y^{(i)}}
\right)
$$

---

# Apply Log Rules

Using:

$$
\log(a^b)=b\log a
$$

we obtain:

$$
\ell(\theta)
=
\sum_{i=1}^{m}
\left[
y^{(i)}\log h_\theta(x^{(i)})
+
(1-y^{(i)})
\log(1-h_\theta(x^{(i)}))
\right]
$$

This is the final log-likelihood equation.

---

# Why Log-Likelihood is Better

Log-likelihood:

- converts products into sums,
- improves numerical stability,
- simplifies differentiation,
- makes optimization easier.

---

# Optimization Objective

We now maximize:

$$
\ell(\theta)
$$

instead of:

$$
L(\theta)
$$

Both produce the same optimal parameters because:

$$
\log(x)
$$

is monotonic.

---

# Relationship to Cost Functions

Linear Regression:

- minimizes squared error.

Logistic Regression:

- maximizes log-likelihood.

This is a fundamental difference.

---

# Concavity of Log-Likelihood

The log-likelihood function is concave.

This is important because:

- there exists one global maximum,
- optimization becomes easier.

---

# Intuition Behind Maximum Likelihood

Good parameters should make observed data:

> highly probable.

Maximum Likelihood Estimation formalizes this intuition mathematically.

---

# Why This Matters in ML

Likelihood-based learning appears throughout Machine Learning:

- Logistic Regression
- Bayesian Learning
- Neural Networks
- Probabilistic Graphical Models

It is one of the most important ideas in statistical learning.

---

# Key Insights

- Likelihood measures how well parameters explain data.
- Products become difficult to optimize.
- Logarithms convert products into sums.
- Logistic Regression optimizes log-likelihood.
- Maximum likelihood is foundational in ML.

---

# Common Mistakes

- Confusing probability with likelihood
- Forgetting independence assumption
- Ignoring why logs are taken
- Mixing likelihood maximization with cost minimization

---

# Implementation Perspective

Modern ML libraries compute:

- probabilities vectorized,
- log-likelihood efficiently,
- gradients automatically during optimization.
