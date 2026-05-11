# GLM Training Intuition

## Motivation

Generalized Linear Models (GLMs) combine:

- linear models,
- probability distributions,
- expected value prediction.

Understanding the training flow is extremely important because it connects:

- features,
- probabilities,
- optimization,
- predictions.

---

# High-Level GLM Pipeline

The complete GLM pipeline can be summarized as:

$$
x
\rightarrow
\theta^Tx
\rightarrow
\eta
\rightarrow
P(y|x)
\rightarrow
\mathbb{E}[y|x]
$$

This pipeline explains how GLMs transform inputs into predictions.

---

# Step 1 — Input Features

Training begins with feature vectors:

$$
x=
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
$$

These features describe the input data.

Examples:

- study hours,
- house size,
- medical measurements.

---

# Step 2 — Linear Combination

The model computes:

$$
\eta=\theta^Tx
$$

This creates a weighted combination of features.

The parameters:

$$
\theta
$$

determine feature importance.

---

# Step 3 — Apply Distribution Assumption

The model assumes outputs follow an exponential family distribution.

Examples:

| Problem | Distribution |
|---|---|
| Continuous outputs | Gaussian |
| Binary outputs | Bernoulli |
| Count outputs | Poisson |

This assumption determines:

- likelihood structure,
- optimization behavior,
- prediction interpretation.

---

# Step 4 — Apply Link Function

The link function transforms:

$$
\eta
$$

into a valid expected output.

Example for Logistic Regression:

$$
\mu=
\frac{1}{1+e^{-\eta}}
$$

Now predictions become valid probabilities.

---

# Step 5 — Compute Expected Value

The final prediction becomes:

$$
\mathbb{E}[y|x]
$$

This represents:

> expected output conditioned on input.

---

# Why Expected Value is Important

Predictions are not treated as exact deterministic outputs.

Instead:

- predictions represent probabilistic expectations,
- uncertainty becomes part of modeling.

---

# Training Objective

The goal of training is to find parameters:

$$
\theta
$$

that maximize likelihood of observed data.

This usually involves:

- Gradient Descent,
- Newton’s Method,
- likelihood optimization.

---

# Why Optimization Works

Exponential family distributions provide:

- differentiable likelihoods,
- smooth optimization surfaces,
- convenient gradients.

This makes learning tractable.

---

# Example — Logistic Regression

Pipeline:

$$
x
\rightarrow
\theta^Tx
\rightarrow
\text{sigmoid}
\rightarrow
P(y=1|x)
$$

Training adjusts parameters so probabilities match observed labels.

---

# Example — Linear Regression

Pipeline:

$$
x
\rightarrow
\theta^Tx
\rightarrow
\mathbb{E}[y|x]
$$

Gaussian assumptions naturally produce squared error optimization.

---

# Intuition Behind Learning

Training repeatedly asks:

> how should parameters change to make observed data more probable?

Optimization gradually improves parameter values.

---

# Why GLMs are Elegant

GLMs unify:

- prediction,
- probability,
- optimization,
- statistics

within one coherent framework.

---

# Relationship to Neural Networks

Neural Networks can be viewed as:

- nonlinear extensions of GLMs,
- stacked transformations of linear predictors.

Many deep learning output layers still use GLM principles.

---

# Key Insights

- GLMs transform features into probabilistic predictions.
- Expected values represent predictions.
- Probability distributions guide optimization.
- Training maximizes likelihood.

---

# Common Mistakes

- Treating GLMs as purely algebraic
- Ignoring probability assumptions
- Forgetting expected value interpretation
- Confusing linear predictor with final prediction

---

# Implementation Perspective

Modern ML systems heavily rely on GLM concepts internally.

Understanding the training flow provides deeper intuition for:

- regression,
- classification,
- probabilistic learning,
- statistical optimization.
