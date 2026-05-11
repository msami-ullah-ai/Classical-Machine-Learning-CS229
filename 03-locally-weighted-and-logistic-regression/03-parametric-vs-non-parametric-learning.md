# Parametric vs Non-Parametric Learning

## Why This Distinction Matters

Machine Learning algorithms can be broadly divided into:

- parametric methods,
- non-parametric methods.

Understanding this distinction is extremely important because it affects:

- memory usage,
- computational cost,
- flexibility,
- scalability.

---

# Parametric Learning Algorithms

## Core Idea

Parametric algorithms assume that data follows some specific mathematical form.

The algorithm learns a fixed set of parameters from training data.

After training:

- the original training data is no longer needed,
- predictions are made using learned parameters only.

---

# Characteristics of Parametric Models

## Fixed Number of Parameters

The number of parameters does not grow with dataset size.

---

## Compresses Information

Training data is summarized into parameter values.

---

## Faster Predictions

Prediction phase becomes efficient because only parameters are needed.

---

# Examples of Parametric Models

Examples include:

- Linear Regression
- Logistic Regression
- Neural Networks

---

# Linear Regression as a Parametric Model

Linear Regression learns:

$$
\theta_0,\theta_1,\ldots,\theta_n
$$

After learning these parameters:

- original training examples can be discarded,
- predictions use only parameter values.

---

# Advantages of Parametric Learning

## Efficient Predictions

Prediction phase is usually fast.

---

## Lower Memory Usage

No need to store entire dataset after training.

---

## Scalable

Often works better for very large datasets.

---

# Disadvantages of Parametric Learning

## Strong Assumptions

The model assumes a specific shape for data.

If assumptions are wrong:

- model performance suffers,
- underfitting may occur.

---

## Limited Flexibility

Fixed parameter structure may struggle with highly complex patterns.

---

# Non-Parametric Learning Algorithms

## Core Idea

Non-parametric algorithms do not assume a fixed mathematical form.

Instead:

- complexity can grow with training data,
- training examples remain important during prediction.

---

# Characteristics of Non-Parametric Models

## Flexible Complexity

Model complexity increases with data size.

---

## Heavy Dependence on Data

Predictions rely strongly on stored examples.

---

## More Flexible

Can model complicated nonlinear relationships.

---

# Examples of Non-Parametric Models

Examples include:

- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Locally Weighted Linear Regression (LWLR)

---

# LWLR as a Non-Parametric Model

LWLR stores training examples and:

- computes local models during prediction,
- adapts based on nearby points.

The model effectively grows with the dataset.

---

# Computational Tradeoff

## Parametric Models

Training may be expensive.

Prediction is usually fast.

---

## Non-Parametric Models

Training may be simple.

Prediction becomes computationally expensive.

---

# Memory Tradeoff

## Parametric Models

Require smaller memory after training.

---

## Non-Parametric Models

Need to store large amounts of training data.

---

# Flexibility Tradeoff

## Parametric Models

Less flexible but more efficient.

---

## Non-Parametric Models

More flexible but more computationally expensive.

---

# Intuition Using Real-World Example

Imagine learning how to identify animals.

---

## Parametric Approach

Learn a few general rules:

- cats have whiskers,
- dogs bark,
- birds have wings.

Compressed knowledge.

---

## Non-Parametric Approach

Memorize many individual examples.

New predictions compare against stored examples.

---

# Choosing Between Them

The choice depends on:

- dataset size,
- computational resources,
- complexity of data,
- desired flexibility.

---

# Key Insights

- Parametric models learn fixed parameters.
- Non-parametric models grow with data.
- Parametric models are efficient but less flexible.
- Non-parametric models are flexible but expensive.

---

# Common Mistakes

- Thinking non-parametric means no parameters
- Assuming flexible models are always better
- Ignoring computational cost
- Confusing training cost with prediction cost

---

# Implementation Perspective

Modern ML systems often balance:

- efficiency,
- flexibility,
- scalability.

Understanding this tradeoff is foundational for selecting appropriate algorithms.
