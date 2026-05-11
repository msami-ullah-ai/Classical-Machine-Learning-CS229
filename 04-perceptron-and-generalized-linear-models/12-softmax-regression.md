# Softmax Regression

## Motivation

Logistic Regression works well for:

> binary classification.

But many real-world problems contain:

> more than two classes.

Examples:

| Problem | Classes |
|---|---|
| Digit Recognition | 0–9 |
| Animal Classification | Cat / Dog / Bird |
| Language Detection | English / Urdu / Arabic |

We therefore need a multiclass extension of Logistic Regression.

This leads to:

> Softmax Regression.

---

# Multiclass Classification

Suppose we have:

$$
K
$$

possible classes.

The model must predict:

> probability of each class.

---

# Core Idea of Softmax

Softmax Regression:

1. computes a score for every class,
2. exponentiates scores,
3. normalizes probabilities.

The final probabilities always sum to 1.

---

# Linear Scores

For each class:

$$
z_i=\theta_i^Tx
$$

Where:

- $\theta_i$ = parameter vector for class $i$.

---

# Why Exponentials Are Used

Exponentials ensure:

$$
e^{z_i}>0
$$

Therefore:
- probabilities remain positive.

Larger scores produce exponentially larger probabilities.

---

# Softmax Function

The Softmax probability for class:

$$
i
$$

is:

$$
P(y=i|x)
=
\frac{e^{\theta_i^Tx}}
{\sum_{k=1}^K e^{\theta_k^Tx}}
$$

---

# Understanding the Formula

The numerator:

$$
e^{\theta_i^Tx}
$$

measures strength of class:

$$
i
$$

The denominator:

$$
\sum_{k=1}^K e^{\theta_k^Tx}
$$

normalizes all probabilities.

---

# Why Probabilities Sum to 1

Since every probability is divided by the same denominator:

$$
\sum_{i=1}^K P(y=i|x)=1
$$

This creates a valid probability distribution.

---

# Intuition Behind Softmax

Softmax converts arbitrary scores into:

> probability distribution over classes.

Higher scores:
- larger probabilities.

Lower scores:
- smaller probabilities.

---

# Example

Suppose scores are:

$$
[2,1,0]
$$

Exponentials:

$$
[e^2,e^1,e^0]
$$

Approximate values:

$$
[7.39,2.72,1]
$$

Normalize:

$$
P=
\left[
\frac{7.39}{11.11},
\frac{2.72}{11.11},
\frac{1}{11.11}
\right]
$$

Result:

$$
[0.665,0.245,0.090]
$$

The first class receives highest probability.

---

# Relationship to Logistic Regression

Softmax Regression generalizes Logistic Regression.

---

## Logistic Regression

Two classes.

---

## Softmax Regression

Multiple classes.

---

# Decision Rule

Prediction becomes:

$$
\arg\max_i P(y=i|x)
$$

Choose the class with highest probability.

---

# Why Softmax is Differentiable

Softmax is smooth and differentiable.

This is critical for:

- Gradient Descent,
- Neural Networks,
- backpropagation.

---

# Relationship to Neural Networks

Softmax is widely used in:

- multiclass neural network outputs,
- image classification,
- NLP models.

It is one of the most important activation functions in deep learning.

---

# Why Softmax is Powerful

Softmax provides:

- probabilistic outputs,
- normalized predictions,
- multiclass capability.

---

# Key Insights

- Softmax extends Logistic Regression to multiple classes.
- Probabilities always sum to 1.
- Exponentials create positive normalized probabilities.
- Highest probability determines prediction.

---

# Common Mistakes

- Confusing Softmax with sigmoid
- Forgetting normalization
- Assuming outputs are independent
- Ignoring probabilistic interpretation

---

# Implementation Perspective

Modern deep learning frameworks heavily use Softmax for:

- image classification,
- language modeling,
- multiclass prediction systems.
