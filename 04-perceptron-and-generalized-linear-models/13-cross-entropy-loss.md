# Cross-Entropy Loss

## Motivation

Classification models predict probabilities.

We therefore need a loss function that measures:

> how well predicted probabilities match true labels.

This leads to:

> Cross-Entropy Loss.

---

# Why Squared Error is Problematic

For classification problems:

- squared error often learns slowly,
- optimization becomes less stable,
- probabilistic interpretation weakens.

Cross-entropy works much better for probability prediction.

---

# Core Idea

Cross-entropy measures:

> difference between true probability distribution and predicted distribution.

Good predictions:
- low loss.

Bad predictions:
- high loss.

---

# Binary Cross-Entropy Loss

For binary classification:

$$
L(y,\hat y)
=
-
\left[
y\log(\hat y)
+
(1-y)\log(1-\hat y)
\right]
$$

Where:

- $y$ = true label
- $\hat y$ = predicted probability

---

# Understanding the Formula

---

## Case 1 — When $y=1$

Loss becomes:

$$
L=-\log(\hat y)
$$

If prediction is close to:

$$
1
$$

loss becomes very small.

If prediction is close to:

$$
0
$$

loss becomes extremely large.

---

## Case 2 — When $y=0$

Loss becomes:

$$
L=-\log(1-\hat y)
$$

Correct probabilities produce small loss.

Wrong confident predictions produce very large penalties.

---

# Why Cross-Entropy Works Well

Cross-entropy strongly penalizes:

> confident incorrect predictions.

This encourages models to learn calibrated probabilities.

---

# Relationship to Maximum Likelihood

Cross-entropy is directly connected to:

> negative log-likelihood.

Minimizing cross-entropy is equivalent to:

> maximizing likelihood.

This gives the loss strong probabilistic foundations.

---

# Multiclass Cross-Entropy

For Softmax Regression:

$$
L
=
-
\sum_{i=1}^K
y_i\log(\hat y_i)
$$

Where:

- $y_i$ = true class indicator
- $\hat y_i$ = predicted probability

---

# Why One-Hot Labels Are Used

True labels are often represented using:

> one-hot vectors.

Example:

Class 2 among 4 classes:

$$
[0,1,0,0]
$$

Only the correct class contributes to the loss.

---

# Intuition Behind the Penalty

Suppose correct class probability is:

$$
0.99
$$

Loss is very small.

Suppose correct class probability is:

$$
0.01
$$

Loss becomes extremely large.

The model is punished heavily for being confidently wrong.

---

# Why Cross-Entropy is Differentiable

Cross-entropy is smooth and differentiable.

This allows efficient optimization using:

- Gradient Descent,
- backpropagation,
- Neural Network training.

---

# Relationship to Information Theory

Cross-entropy originally comes from:

> information theory.

It measures:

- uncertainty,
- information mismatch,
- probability divergence.

---

# Why Deep Learning Uses Cross-Entropy

Cross-entropy is widely used because it:

- optimizes probabilities effectively,
- works well with Softmax,
- provides strong gradients,
- trains neural networks efficiently.

---

# Example

Suppose true label:

$$
y=1
$$

Prediction:

$$
\hat y=0.9
$$

Loss:

$$
L=-\log(0.9)
$$

Small loss.

---

Suppose:

$$
\hat y=0.01
$$

Loss:

$$
L=-\log(0.01)
$$

Very large loss.

---

# Key Insights

- Cross-entropy measures probability mismatch.
- Wrong confident predictions are heavily penalized.
- Cross-entropy is equivalent to negative log-likelihood.
- It is foundational in modern classification systems.

---

# Common Mistakes

- Using squared error for classification
- Ignoring probabilistic interpretation
- Forgetting one-hot encoding
- Confusing probabilities with logits

---

# Implementation Perspective

Modern ML frameworks heavily rely on cross-entropy for:

- Logistic Regression,
- Softmax Regression,
- Neural Networks,
- deep learning classification systems.
