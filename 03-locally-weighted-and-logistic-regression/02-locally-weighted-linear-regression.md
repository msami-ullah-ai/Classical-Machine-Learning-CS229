# Locally Weighted Linear Regression (LWLR)

## Motivation

Standard Linear Regression learns:

> one global model for the entire dataset.

This becomes problematic when:

- data is nonlinear,
- different regions behave differently,
- one straight line cannot fit all patterns.

Locally Weighted Linear Regression solves this problem using local learning.

---

# Core Idea of LWLR

Instead of fitting:

> one global linear model,

LWLR fits:

> a local linear model around each prediction point.

For every new query point:

- nearby examples receive high importance,
- distant examples receive low importance.

---

# Local Learning Intuition

Suppose we want prediction near some point:

$$
x
$$

Nearby data points are more relevant because:

- they better represent local behavior,
- they contain similar patterns.

Faraway points may belong to completely different regions.

---

# Weighted Linear Regression

LWLR minimizes a weighted loss function:

$$
\sum_{i=1}^{m}
w^{(i)}
(y^{(i)}-\theta^Tx^{(i)})^2
$$

Where:

- $w^{(i)}$ = weight of training example
- nearby points get larger weights
- distant points get smaller weights

---

# How LWLR Works

For each prediction point:

1. compute weights for all training examples,
2. assign high importance to nearby points,
3. fit a local linear model,
4. make prediction.

This process repeats for every query point.

---

# Important Intuition

A key idea in LWLR:

> nearby points matter more than distant points.

This allows the model to adapt locally instead of forcing one global behavior.

---

# Why LWLR Handles Nonlinear Data

Although each local model is linear:

- different local regions have different fitted lines.

Combining many local linear fits creates:

> an overall nonlinear behavior.

This is why LWLR can model curved patterns effectively.

---

# Local Linear Approximation

LWLR assumes:

- very small local regions can often be approximated linearly.

Globally:
- data may be nonlinear.

Locally:
- data may still look approximately linear.

---

# Flexible Behavior

Unlike ordinary Linear Regression:

- LWLR changes behavior depending on query location.

Predictions near different regions may use completely different local models.

---

# Prediction Process

Suppose we want prediction at:

$$
x^{(query)}
$$

LWLR:

- computes distances between query point and training examples,
- assigns weights,
- fits weighted regression,
- predicts output.

---

# Advantages of LWLR

## Handles Nonlinear Patterns

Can model curved relationships effectively.

---

## Local Adaptation

Different regions can behave differently.

---

## Flexible Model

No single global line is forced onto all data.

---

# Disadvantages of LWLR

## Computationally Expensive

A new local model must be fit for every prediction.

---

## Memory Intensive

Training data must be stored.

---

## Slow Predictions

Prediction time becomes expensive for large datasets.

---

# Why LWLR is Non-Parametric

LWLR does not compress data into fixed parameters.

Instead:

- training examples remain important during prediction,
- model complexity grows with dataset size.

This makes LWLR a non-parametric algorithm.

---

# Real-World Intuition

Imagine predicting house prices.

Nearby houses:

- are more relevant,
- share similar neighborhoods,
- have similar market conditions.

Faraway houses may be irrelevant.

LWLR naturally captures this intuition.

---

# Relationship to Kernel Methods

LWLR uses weighted influence based on distance.

This idea later appears in:

- kernel methods,
- radial basis functions,
- support vector machines,
- Gaussian processes.

---

# Key Insights

- LWLR fits local linear models.
- Nearby points receive larger importance.
- Different regions can behave differently.
- Local learning helps model nonlinear data.

---

# Common Mistakes

- Thinking LWLR learns one global model
- Ignoring computational cost
- Assuming local models are globally linear
- Forgetting weights change for every query point

---

# Implementation Perspective

In practice:

- distances are computed for all training examples,
- weights are assigned using Gaussian functions,
- weighted least squares is solved repeatedly.

LWLR becomes expensive for large datasets but works very well for low-dimensional problems.
