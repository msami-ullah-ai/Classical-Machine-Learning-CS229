# Hypothesis Function and Parameters

## Understanding the Hypothesis Function

The hypothesis function is the mathematical function used by Linear Regression to make predictions.

For a single feature:

$$
h_\theta(x)=\theta_0+\theta_1x
$$

This equation represents a straight line.

---

## General Form of Linear Regression

For multiple features:

$$
h_\theta(x)=\sum_{j=0}^{n}\theta_jx_j
$$

Expanded form:

$$
h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+\cdots+\theta_nx_n
$$

---

## Understanding Parameters

The values:

$$
\theta_0,\theta_1,\theta_2,\ldots,\theta_n
$$

are called parameters or weights.

They determine:

- how important each feature is,
- how strongly each feature affects predictions.

---

## Role of Each Parameter

| Parameter | Purpose |
|---|---|
| $\theta_0$ | Bias / intercept |
| $\theta_1$ | Weight of feature 1 |
| $\theta_2$ | Weight of feature 2 |
| ... | ... |

---

## Intuition Behind Weights

Weights determine feature importance.

Large positive weight:
- strong positive effect

Negative weight:
- inverse relationship

Weight near zero:
- weak contribution

---

## Bias Term ($\theta_0$)

The bias term is extremely important.

Without bias:

$$
h_\theta(x)=\theta_1x
$$

the line is forced to pass through the origin.

That reduces model flexibility.

---

## Why Bias Matters

Suppose we predict house prices.

Even if:

- house size = 0,
- bedrooms = 0,

there may still be some base price.

The bias term captures this baseline value.

---

## Feature Vector Representation

To simplify notation, we define:

$$
x_0=1
$$

Now the feature vector becomes:

$$
x=[1,x_1,x_2,\ldots,x_n]
$$

This allows bias to become part of the dot product.

Now the hypothesis becomes:

$$
h_\theta(x)=\theta^Tx
$$

---

## Example

Suppose:

$$
x=[1,2000,3]
$$

Where:

- 1 = bias feature
- 2000 = house size
- 3 = number of bedrooms

And:

$$
\theta=[50,100,20]
$$

Prediction:

$$
h_\theta(x)=50+100(2000)+20(3)
$$

---

## Training Examples

Each training example is represented as:

$$
(x^{(i)},y^{(i)})
$$

Example:

| $x^{(i)}$ | $y^{(i)}$ |
|---|---|
| [1,1200,2] | 180 |
| [1,2000,4] | 350 |

---

## Number of Features vs Number of Examples

| Symbol | Meaning |
|---|---|
| $m$ | Number of training examples |
| $n$ | Number of features |

Important distinction:

- rows = training examples
- columns = features

---

## Why Feature Representation Matters

A good feature representation helps the model:

- learn patterns more effectively,
- make better predictions,
- train faster.

Poor feature representation often leads to weak models.

---

## Key Insights

- Parameters control predictions.
- Bias improves flexibility.
- Features are inputs to the model.
- Hypothesis combines features linearly.

---

## Common Mistakes

- Forgetting $x_0=1$
- Mixing parameters with features
- Ignoring the importance of bias
- Misunderstanding weight interpretation

---

## Implementation Perspective

In NumPy:

- parameters are vectors,
- features are matrices,
- predictions are computed using dot products.

Vectorization makes training very fast for large datasets.
