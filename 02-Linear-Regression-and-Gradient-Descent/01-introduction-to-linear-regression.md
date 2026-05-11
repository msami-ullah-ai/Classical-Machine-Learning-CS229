# Introduction to Linear Regression

## What is Machine Learning?

Machine Learning is the field of building algorithms that learn patterns from data instead of being explicitly programmed with fixed rules.

A learning algorithm:

- takes a training dataset as input,
- learns relationships from the data,
- outputs a mathematical function called a hypothesis.

The purpose of the hypothesis is to make predictions on unseen examples.

---

## The Goal of a Learning Algorithm

The main job of a learning algorithm is:

- find patterns in data,
- learn relationships between variables,
- make predictions as accurately as possible.

We usually do not know the true real-world relationship between variables.

Instead, the model tries to approximate it using mathematical functions.

---

## Supervised Learning

Linear Regression belongs to **Supervised Learning**.

In supervised learning:

- inputs and outputs are already given,
- the algorithm learns the mapping between them.

Example:

| House Size | House Price |
|---|---|
| 1000 sq ft | \$150,000 |
| 1500 sq ft | \$220,000 |
| 2000 sq ft | \$300,000 |

Here:

- input = house size
- output = house price

The model learns how input affects output.

---

## What is Linear Regression?

Linear Regression is one of the simplest and most important machine learning algorithms.

It models the relationship between inputs and outputs using a straight line.

The algorithm tries to find the best fitting line through the training data.

---

## Hypothesis Function

The prediction function used by Linear Regression is called the **hypothesis function**.

For a single feature:

$$
h_\theta(x)=\theta_0+\theta_1x
$$

Where:

- $x$ = input feature
- $h_\theta(x)$ = predicted value
- $\theta_0$ = bias/intercept
- $\theta_1$ = weight/slope

---

## Intuition Behind the Hypothesis

The model predicts outputs using a line.

Example:

- if house size increases,
- predicted price should also increase.

The parameters $\theta_0$ and $\theta_1$ determine:

- where the line starts,
- how steep the line becomes.

The learning algorithm adjusts these parameters during training.

---

## Parameters of the Model

The parameters are:

$$
\theta_0,\theta_1,\theta_2,\ldots,\theta_n
$$

These values control model behavior.

The learning algorithm continuously updates them to reduce prediction error.

---

## Training Examples

Each row in the dataset is called a **training example**.

Notation:

$$
(x^{(i)},y^{(i)})
$$

Where:

- $x^{(i)}$ = input features of the $i^{th}$ example
- $y^{(i)}$ = actual target value

---

## Important Notation

| Symbol | Meaning |
|---|---|
| $m$ | Number of training examples |
| $n$ | Number of features |
| $x$ | Input feature vector |
| $y$ | Actual output |
| $h_\theta(x)$ | Predicted output |
| $\theta$ | Model parameters |

---

## Feature Representation

For multiple features:

$$
x=[x_1,x_2,x_3,\ldots,x_n]
$$

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

## Learning From Data

The hypothesis is not the true real-world function.

It is only the model's best approximation learned from training data.

Better parameters lead to better predictions.

---

## Goal of Linear Regression

The ultimate goal is:

- learn relationships from data,
- minimize prediction error,
- make predictions close to actual values.

---

## Key Insights

- Machine Learning learns patterns from data.
- Linear Regression uses straight lines for prediction.
- Parameters control model behavior.
- Training examples teach the model relationships.

---

## Common Mistakes

- Confusing features with parameters
- Thinking the hypothesis is always correct
- Forgetting that models only approximate reality
- Assuming all relationships are perfectly linear

---

## Implementation Perspective

In practice:

- features are stored in matrices,
- parameters are stored in vectors,
- predictions are computed using matrix multiplication.

Libraries like NumPy make these computations extremely efficient.
