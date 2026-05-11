# Limitations of Linear Regression

## Why Linear Regression is Not Always Enough

Linear Regression assumes that the relationship between input features and outputs is approximately linear.

The model tries to fit a straight line through the data.

However, real-world data is often much more complex.

Many real-world patterns:

- are curved,
- contain nonlinear relationships,
- cannot be represented accurately using a single straight line.

---

# Linear Regression Assumption

Linear Regression assumes:

$$
h_\theta(x)=\theta_0+\theta_1x
$$

This equation represents a straight line.

The model works well only when the underlying relationship is close to linear.

---

# Problem with Nonlinear Data

Suppose data follows a curved pattern.

A single straight line may:

- fail to capture the true relationship,
- produce large prediction errors,
- underfit the dataset.

---

# Underfitting

Underfitting occurs when:

- the model is too simple,
- it cannot capture important patterns in the data.

Example:

- curved data,
- but model uses only a straight line.

The model misses important structure.

---

# Real-World Data is Often Nonlinear

Examples of nonlinear relationships:

| Domain | Nonlinear Pattern |
|---|---|
| Population Growth | Exponential |
| Temperature Trends | Cyclical |
| Biological Systems | Complex nonlinear behavior |
| Financial Markets | Irregular patterns |

A single linear model may struggle with such data.

---

# Why Global Linear Models Fail

Standard Linear Regression learns:

> one global model for the entire dataset.

This means:

- all points are treated equally,
- one line attempts to explain everything.

For complex datasets:

- different regions may behave differently.

A single global model becomes insufficient.

---

# Local Patterns in Data

Many datasets contain local behavior.

Example:

- one region may increase rapidly,
- another region may flatten,
- another may decrease.

A single straight line cannot adapt locally.

---

# Motivation for Local Models

Instead of fitting:

> one global linear model,

we can fit:

> local models around nearby data points.

This idea leads to:

> Locally Weighted Linear Regression (LWLR)

---

# Intuition Behind Local Learning

Nearby points are often more relevant than distant points.

For a prediction near some location:

- nearby examples should influence prediction strongly,
- faraway examples should influence prediction weakly.

LWLR uses this intuition.

---

# From Global Learning to Local Learning

## Standard Linear Regression

- one model,
- fixed parameters,
- same behavior everywhere.

---

## Locally Weighted Regression

- different local model for each prediction point,
- adapts to local patterns,
- handles nonlinear data more effectively.

---

# Why This Matters in Machine Learning

Many advanced ML algorithms are designed because:

> simple global linear assumptions often fail in practice.

Understanding the limitations of Linear Regression helps motivate:

- local learning,
- nonlinear models,
- kernel methods,
- neural networks.

---

# Key Insights

- Linear Regression assumes linear relationships.
- Real-world data is often nonlinear.
- Global linear models may underfit complex data.
- Local models can adapt better to different regions.

---

# Common Mistakes

- Assuming Linear Regression works for every dataset
- Ignoring nonlinear patterns
- Using overly simple models for complex relationships
- Confusing underfitting with overfitting

---

# Implementation Perspective

When Linear Regression performs poorly:

- feature engineering,
- polynomial features,
- local learning methods,
- or nonlinear models

may improve performance significantly.
