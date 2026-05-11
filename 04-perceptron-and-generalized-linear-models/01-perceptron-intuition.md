# Perceptron Intuition

## What is the Perceptron?

The Perceptron is one of the earliest Machine Learning algorithms developed for binary classification.

Its goal is simple:

> find a linear decision boundary that separates data into two classes.

It is historically important because it introduced many ideas later used in Neural Networks.

---

# Binary Classification

Perceptron works on binary classification problems.

The output belongs to one of two classes:

$$
y \in \{0,1\}
$$

Examples:

| Problem | Classes |
|---|---|
| Spam Detection | Spam / Not Spam |
| Tumor Detection | Malignant / Benign |
| Exam Result | Pass / Fail |

---

# Linear Decision Boundary

The Perceptron tries to learn a boundary such as:

$$
\theta^Tx+b=0
$$

This boundary separates data points into two regions.

Points on one side:
- class 1

Points on the other side:
- class 0

---

# Relationship to Logistic Regression

Perceptron and Logistic Regression are closely related.

Both compute:

$$
z=\theta^Tx+b
$$

The difference lies in the activation function used.

---

# Logistic Regression Uses Sigmoid

Logistic Regression uses:

$$
g(z)=\frac{1}{1+e^{-z}}
$$

This produces probabilities between:

$$
0 \le g(z) \le 1
$$

---

# Perceptron Uses Step Function

Instead of sigmoid, Perceptron uses a hard threshold function:

$$
g(z)=
\begin{cases}
1 & z \ge 0 \\
0 & z < 0
\end{cases}
$$

This is called the:

> Step Function.

---

# Intuition Behind the Step Function

The step function behaves like a switch.

If:

$$
z \ge 0
$$

predict:

$$
y=1
$$

Otherwise:

$$
y=0
$$

The output changes abruptly at zero.

---

# Hard Classification

Perceptron makes:

> hard decisions.

It directly predicts class labels.

Unlike Logistic Regression:
- no probabilities are produced.

---

# Visual Intuition

Imagine drawing a straight line through data points.

Goal:

- positives on one side,
- negatives on the other side.

The Perceptron continuously adjusts the line until classification improves.

---

# Why Perceptron Was Important

The Perceptron introduced several foundational ideas:

- linear decision boundaries,
- weighted inputs,
- activation functions,
- iterative learning.

These ideas later evolved into modern Neural Networks.

---

# Inputs and Weights

Each input feature has an associated weight.

The model computes:

$$
z=\theta_1x_1+\theta_2x_2+\cdots+\theta_nx_n+b
$$

The weights determine:

- feature importance,
- boundary orientation.

---

# Geometric Interpretation

The equation:

$$
\theta^Tx+b=0
$$

defines a hyperplane.

In:

- 2D → line
- 3D → plane
- higher dimensions → hyperplane

This hyperplane separates classes.

---

# Why Perceptron Works

If data is linearly separable:

- there exists some boundary,
- Perceptron can eventually find it.

This is one of the most important theoretical properties of the algorithm.

---

# Difference Between Perceptron and Logistic Regression

| Perceptron | Logistic Regression |
|---|---|
| Hard outputs | Probabilistic outputs |
| Step function | Sigmoid function |
| Non-differentiable | Differentiable |
| Binary labels only | Probability estimation |

---

# Key Insights

- Perceptron is an early binary classifier.
- It learns a linear decision boundary.
- It uses a step activation function.
- Predictions are hard class labels.

---

# Common Mistakes

- Confusing perceptron with logistic regression
- Assuming perceptron outputs probabilities
- Forgetting the role of the step function
- Ignoring linear separability assumptions

---

# Implementation Perspective

In practice:

- perceptrons are rarely used directly today,
- but their core ideas are foundational to deep learning and neural networks.
