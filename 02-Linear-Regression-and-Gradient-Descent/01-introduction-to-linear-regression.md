# Introduction to Linear Regression

## What is Machine Learning?

Machine Learning is the field of building algorithms that learn patterns from data instead of being explicitly programmed with fixed rules.

A learning algorithm:

- takes a **training dataset** as input,
- learns patterns and relationships from that data,
- and outputs a mathematical function called a **hypothesis**.

The goal is to make accurate predictions on unseen examples.

---

## Supervised Learning

Linear Regression belongs to **Supervised Learning**.

In supervised learning:

- we are given input-output pairs,
- the algorithm learns the relationship between them.

Example:

| House Size | House Price |
|---|---|
| 1000 sq ft | \$150,000 |
| 1500 sq ft | \$220,000 |
| 2000 sq ft | \$300,000 |

Here:

- input = house size
- output = house price

The model learns how house size affects house price.

---

## What is Linear Regression?

Linear Regression is one of the simplest and most important machine learning algorithms.

It tries to model the relationship between input features and output values using a straight line.

The model attempts to find the **best fitting line** through the data.

---

## Hypothesis Function

The prediction function used by the model is called the **hypothesis function**.

For simple linear regression:

:contentReference[oaicite:0]{index=0}

Where:

- \(x\) = input feature
- \(h_\theta(x)\) = predicted value
- \(\theta_0\) = bias/intercept
- \(\theta_1\) = weight/slope

---

## Intuition Behind the Hypothesis

The model makes predictions using a line.

Example:

- if house size increases,
- predicted house price should also increase.

The parameters \(\theta_0\) and \(\theta_1\) control:

- where the line starts,
- and how steep the line is.

The learning algorithm's job is to find the best values of these parameters.

---

## Parameters of the Model

The parameters are:

\[
\theta_0, \theta_1, \theta_2, ..., \theta_n
\]

These values determine how the model behaves.

The learning algorithm continuously adjusts them to reduce prediction error.

---

## Training Examples

Each row in the dataset is called a **training example**.

Notation:

\[
(x^{(i)}, y^{(i)})
\]

Where:

- \(x^{(i)}\) = input features of the \(i^{th}\) example
- \(y^{(i)}\) = actual target value

---

## Important Notation

| Symbol | Meaning |
|---|---|
| \(m\) | Number of training examples |
| \(n\) | Number of features |
| \(x\) | Input feature vector |
| \(y\) | Actual output |
| \(h_\theta(x)\) | Predicted output |
| \(\theta\) | Model parameters |

---

## Feature Representation

For multiple features:

\[
x = [x_1, x_2, x_3, ..., x_n]
\]

Example:

| Size | Bedrooms | Price |
|---|---|---|
| 1200 | 2 | \$180k |
| 2000 | 4 | \$350k |

Features:

- size
- bedrooms

Target:

- price

---

## Goal of Linear Regression

The ultimate goal is:

- learn the relationship between inputs and outputs,
- make predictions as close as possible to real values,
- minimize prediction error.

---

## Key Insights

- Linear Regression learns patterns from data.
- The hypothesis function makes predictions.
- Parameters control the behavior of the model.
- Training examples help the model learn relationships.

---

## Common Mistakes

- Confusing features with parameters
- Thinking the hypothesis is always correct
- Forgetting that the model only approximates reality
- Assuming the relationship is always perfectly linear

---

## Implementation Perspective

In practice:

- features are stored in matrices,
- parameters are stored in vectors,
- predictions are computed using matrix multiplication.

Libraries like NumPy make these computations efficient.
