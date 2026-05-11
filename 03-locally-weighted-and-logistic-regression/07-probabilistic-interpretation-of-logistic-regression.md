# Probabilistic Interpretation of Logistic Regression

## Logistic Regression as a Probabilistic Model

Logistic Regression does not directly predict class labels.

Instead, it predicts probabilities.

Specifically:

$$
P(y=1|x;\theta)
$$

which means:

> probability that input $x$ belongs to class 1.

---

# Probability of Class 1

Logistic Regression defines:

$$
P(y=1|x;\theta)=h_\theta(x)
$$

Where:

$$
h_\theta(x)=g(\theta^Tx)
$$

and:

$$
g(z)=\frac{1}{1+e^{-z}}
$$

---

# Probability of Class 0

Since probabilities must sum to 1:

$$
P(y=0|x;\theta)=1-h_\theta(x)
$$

---

# Understanding the Meaning

Suppose:

$$
h_\theta(x)=0.9
$$

This means:

- 90% probability of class 1,
- 10% probability of class 0.

---

# Binary Labels

Recall:

$$
y\in\{0,1\}
$$

Where:

- $y=1$ → positive class
- $y=0$ → negative class

---

# Combining Both Cases into One Equation

Instead of writing separate equations for:

- $y=1$
- $y=0$

we combine them into one compact expression:

$$
P(y|x;\theta)
=
h_\theta(x)^y
(1-h_\theta(x))^{1-y}
$$

This is one of the most important equations in Logistic Regression.

---

# Why This Formula Works

---

## Case 1: When $y=1$

Substitute:

$$
y=1
$$

Result:

$$
P(y|x;\theta)
=
h_\theta(x)^1
(1-h_\theta(x))^0
$$

Since anything raised to power 0 equals 1:

$$
P(y|x;\theta)=h_\theta(x)
$$

Correct.

---

## Case 2: When $y=0$

Substitute:

$$
y=0
$$

Result:

$$
P(y|x;\theta)
=
h_\theta(x)^0
(1-h_\theta(x))^1
$$

Therefore:

$$
P(y|x;\theta)=1-h_\theta(x)
$$

Correct again.

---

# Why This Representation is Powerful

This compact form allows us to:

- write one unified probability equation,
- simplify derivations,
- derive likelihood functions cleanly.

It becomes the foundation for Logistic Regression optimization.

---

# Probabilistic Interpretation of the Hypothesis

The hypothesis is no longer:

> predicted numerical value.

Instead, it becomes:

> predicted probability of belonging to class 1.

This is a major conceptual shift from Linear Regression.

---

# Decision Rule

After computing probability:

$$
h_\theta(x)\ge0.5
$$

predict:

$$
y=1
$$

Otherwise:

$$
y=0
$$

---

# Relationship to Confidence

Probability also measures model confidence.

Example:

| Probability | Interpretation |
|---|---|
| 0.99 | Very confident class 1 |
| 0.51 | Slightly confident class 1 |
| 0.50 | Completely uncertain |
| 0.02 | Very confident class 0 |

---

# Why Probabilities Matter

Probability outputs are useful because they provide:

- uncertainty information,
- confidence estimation,
- interpretable predictions.

This is often more valuable than hard labels alone.

---

# Relationship to Maximum Likelihood

Once probabilities are defined:

- we can evaluate how likely observed data is,
- optimize parameters using likelihood functions.

This leads directly to:

> Maximum Likelihood Estimation (MLE)

---

# Key Insights

- Logistic Regression predicts probabilities.
- Class probabilities sum to 1.
- A unified probability equation simplifies derivations.
- Probabilistic interpretation is central to Logistic Regression.

---

# Common Mistakes

- Confusing probabilities with labels
- Forgetting probabilities must sum to 1
- Misunderstanding the combined probability equation
- Assuming Logistic Regression outputs exact certainty

---

# Implementation Perspective

Modern ML systems:

- compute probabilities vectorized,
- apply thresholding for classification,
- optimize parameters using likelihood maximization.
