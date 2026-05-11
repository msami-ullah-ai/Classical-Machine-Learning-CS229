# Hypothesis Function and Parameters

## Understanding the Hypothesis Function

The hypothesis function is the mathematical function used by Linear Regression to make predictions.

For a single feature:

:contentReference[oaicite:1]{index=1}

This equation represents a straight line.

---

## General Form of Linear Regression

For multiple features:

:contentReference[oaicite:2]{index=2}

Expanded form:

\[
h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+\cdots+\theta_nx_n
\]

---

## Understanding Parameters

The values:

\[
\theta_0,\theta_1,\theta_2,...,\theta_n
\]

are called **parameters** or **weights**.

They determine:

- how important each feature is,
- how strongly each feature affects predictions.

---

## Role of Each Parameter

| Parameter | Purpose |
|---|---|
| \(\theta_0\) | Bias / intercept |
| \(\theta_1\) | Weight of feature 1 |
| \(\theta_2\) | Weight of feature 2 |
| ... | ... |

---

## Bias Term (\(\theta_0\))

The bias term is extremely important.

Without bias:

\[
h_\theta(x)=\theta_1x
\]

the line is forced to pass through the origin.

That reduces model flexibility.

Adding bias allows the model to shift upward or downward for a better fit.

---

## Why Bias Matters

Suppose we predict house prices.

Even if:

- house size = 0,
- bedrooms = 0,

there may still be a base price.

The bias term captures this baseline value.

---

## Feature Vector Representation

To simplify notation, we define:

\[
x_0=1
\]

Then the feature vector becomes:

\[
x=[1,x_1,x_2,...,x_n]
\]

Now the hypothesis becomes:

\[
h_\theta(x)=\theta^Tx
\]

This compact form is heavily used in vectorized implementations.

---

## Example

Suppose:

\[
x=[1,2000,3]
\]

Where:

- 1 = bias feature
- 2000 = house size
- 3 = number of bedrooms

And:

\[
\theta=[50,100,20]
\]

Prediction:

\[
h_\theta(x)=50+100(2000)+20(3)
\]

---

## Training Examples

Each training example is represented as:

\[
(x^{(i)}, y^{(i)})
\]

Example:

| \(x^{(i)}\) | \(y^{(i)}\) |
|---|---|
| [1, 1200, 2] | 180 |
| [1, 2000, 4] | 350 |

---

## Number of Features vs Number of Examples

| Symbol | Meaning |
|---|---|
| \(m\) | Number of training examples |
| \(n\) | Number of features |

Important distinction:

- rows = training examples
- columns = features

---

## Intuition Behind Weights

Weights determine how much each feature contributes to predictions.

Large positive weight:
- strong positive influence

Negative weight:
- inverse relationship

Near zero:
- weak influence

---

## Key Insights

- Parameters control model behavior.
- Bias allows flexible fitting.
- Features are inputs to the model.
- The hypothesis combines features linearly.

---

## Common Mistakes

- Forgetting \(x_0 = 1\)
- Mixing features and parameters
- Assuming larger weight always means better feature
- Ignoring the role of bias

---

## Implementation Perspective

In practice:

- parameters are stored in vectors,
- features are stored in matrices,
- predictions are computed efficiently using dot products.

Vectorization allows very fast computation for large datasets.
