# Worked Example of Logistic Regression

## Goal

In this example, we manually compute:

- linear score,
- sigmoid output,
- predicted probability,
- final classification.

This demonstrates how Logistic Regression works step-by-step.

---

# Problem Setup

Suppose we want to predict whether a student passes an exam.

Feature:

$$
x = \text{study hours}
$$

Output:

$$
y \in \{0,1\}
$$

Where:

- $0$ = fail
- $1$ = pass

---

# Model Parameters

Suppose:

$$
\theta_0=-4
$$

$$
\theta_1=1.5
$$

---

# Hypothesis Function

Logistic Regression hypothesis:

$$
h_\theta(x)=g(\theta^Tx)
$$

Where sigmoid function is:

$$
g(z)=\frac{1}{1+e^{-z}}
$$

---

# Example Input

Suppose a student studies:

$$
x=4
$$

hours.

---

# Step 1 — Compute Linear Score

The linear score is:

$$
z=\theta_0+\theta_1x
$$

Substitute values:

$$
z=-4+1.5(4)
$$

$$
z=-4+6
$$

$$
z=2
$$

---

# Step 2 — Apply Sigmoid Function

Now compute:

$$
g(2)=\frac{1}{1+e^{-2}}
$$

Approximate value:

$$
e^{-2}\approx0.135
$$

Therefore:

$$
g(2)=\frac{1}{1+0.135}
$$

$$
g(2)\approx0.881
$$

---

# Step 3 — Interpret the Probability

The model predicts:

$$
h_\theta(x)\approx0.881
$$

Meaning:

> there is approximately 88.1% probability that the student passes.

---

# Step 4 — Final Classification

Decision rule:

- if probability ≥ 0.5 → predict class 1
- otherwise → predict class 0

Since:

$$
0.881 > 0.5
$$

Prediction becomes:

$$
y=1
$$

The student is predicted to pass.

---

# Another Example

Suppose:

$$
x=1
$$

hour.

---

# Compute Linear Score

$$
z=-4+1.5(1)
$$

$$
z=-2.5
$$

---

# Apply Sigmoid

$$
g(-2.5)=\frac{1}{1+e^{2.5}}
$$

Approximate:

$$
g(-2.5)\approx0.076
$$

---

# Interpretation

The probability of passing is approximately:

$$
7.6\%
$$

Since:

$$
0.076 < 0.5
$$

Prediction becomes:

$$
y=0
$$

The student is predicted to fail.

---

# Understanding the Decision Boundary

The decision boundary occurs where:

$$
h_\theta(x)=0.5
$$

Since:

$$
g(0)=0.5
$$

the boundary occurs when:

$$
\theta^Tx=0
$$

For this example:

$$
-4+1.5x=0
$$

Solve:

$$
x=\frac{4}{1.5}
$$

$$
x\approx2.67
$$

Students studying more than:

$$
2.67
$$

hours are predicted to pass.

---

# What This Example Demonstrates

This example shows the complete Logistic Regression pipeline:

1. compute linear score,
2. apply sigmoid,
3. interpret probability,
4. make classification decision.

---

# Why Logistic Regression is Probabilistic

Unlike Linear Regression:

- Logistic Regression does not predict raw outputs.

Instead:

- it predicts probabilities,
- then converts probabilities into class labels.

---

# Key Insights

- Logistic Regression predicts probabilities.
- Sigmoid converts scores into probabilities.
- Decision boundaries separate classes.
- Classification depends on probability threshold.

---

# Common Mistakes

- Forgetting sigmoid transformation
- Confusing score with probability
- Ignoring thresholding step
- Assuming outputs can exceed 1

---

# Implementation Perspective

In practice:

- predictions are vectorized,
- sigmoid is applied element-wise,
- probabilities are thresholded automatically for classification.
