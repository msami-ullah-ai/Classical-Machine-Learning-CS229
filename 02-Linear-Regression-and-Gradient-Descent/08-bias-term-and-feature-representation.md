# Bias Term and Feature Representation

## Understanding Features

Features are the input variables used by the model to make predictions.

Examples:

| Feature | Meaning |
|---|---|
| House Size | Area of house |
| Bedrooms | Number of bedrooms |
| Age | Age of house |

These inputs help the model learn relationships in the data.

---

# Feature Vector

Instead of writing features separately:

$$
x_1,x_2,x_3,\ldots,x_n
$$

we combine them into a vector:

$$
x=[x_1,x_2,x_3,\ldots,x_n]
$$

This makes mathematical notation cleaner and easier to implement.

---

# Hypothesis Using Multiple Features

For multiple features:

$$
h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+\cdots+\theta_nx_n
$$

Each feature contributes to the final prediction.

---

# Understanding the Bias Term

The parameter:

$$
\theta_0
$$

is called the bias or intercept term.

It represents the model's baseline prediction.

---

# Why Bias is Important

Without bias:

$$
h_\theta(x)=\theta_1x_1+\theta_2x_2+\cdots+\theta_nx_n
$$

the model is forced to pass through the origin.

That severely reduces flexibility.

---

# Intuition Behind the Bias

Suppose we predict house prices.

Even if:

- house size = 0,
- bedrooms = 0,

there may still be some base price.

The bias captures this baseline value.

---

# Adding the Bias Feature

To simplify equations, we define:

$$
x_0=1
$$

Now the feature vector becomes:

$$
x=[1,x_1,x_2,\ldots,x_n]
$$

This allows the bias to become part of the dot product.

---

# Compact Hypothesis Representation

The hypothesis can now be written compactly as:

$$
h_\theta(x)=\theta^Tx
$$

Where:

$$
\theta=
[\theta_0,\theta_1,\theta_2,\ldots,\theta_n]
$$

and:

$$
x=[1,x_1,x_2,\ldots,x_n]
$$

---

# Why This Representation is Powerful

This representation:

- simplifies notation,
- enables vectorization,
- makes matrix operations easier.

Modern ML implementations heavily rely on this format.

---

# Example

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

# Understanding Feature Importance

Each parameter controls the influence of its corresponding feature.

Large positive weight:
- strong positive effect

Negative weight:
- inverse relationship

Near-zero weight:
- weak effect

---

# Number of Features vs Number of Examples

| Symbol | Meaning |
|---|---|
| $m$ | Number of training examples |
| $n$ | Number of features |

Important distinction:

- rows = examples
- columns = features

---

# Feature Engineering

The quality of features strongly affects model performance.

Good features:
- improve learning,
- improve predictions,
- reduce error.

Poor features:
- limit model capability.

---

# Why Feature Representation Matters in ML

Feature representation affects:

- learning speed,
- optimization quality,
- model accuracy.

In many real-world ML systems:

> feature engineering is as important as the algorithm itself.

---

# Key Insights

- Features are model inputs.
- Bias improves model flexibility.
- Feature vectors simplify notation.
- Dot product representation enables vectorization.

---

# Common Mistakes

- Forgetting $x_0=1$
- Mixing features and parameters
- Ignoring the role of bias
- Assuming all features contribute equally

---

# Implementation Perspective

In practice:

- features are stored in matrices,
- parameters are vectors,
- predictions use efficient matrix multiplication.

This representation is foundational for modern machine learning systems.
