# Perceptron Update Rule

## Goal of Learning

The Perceptron learns by adjusting its parameters whenever it makes a mistake.

If a point is classified correctly:
- no update is needed.

If a point is classified incorrectly:
- weights are adjusted.

This iterative correction process gradually improves the decision boundary.

---

# Perceptron Prediction

The Perceptron computes:

$$
z=\theta^Tx+b
$$

Then applies the step function:

$$
\hat y=
\begin{cases}
1 & z \ge 0 \\
0 & z < 0
\end{cases}
$$

Where:

- $\hat y$ = predicted label

---

# Perceptron Learning Rule

The update rule is:

$$
\theta_j
=
\theta_j
+
\alpha
(y^{(i)}-\hat y^{(i)})x_j^{(i)}
$$

Where:

- $\alpha$ = learning rate
- $y^{(i)}$ = actual label
- $\hat y^{(i)}$ = predicted label
- $x_j^{(i)}$ = feature value

---

# Understanding the Error Term

The term:

$$
y^{(i)}-\hat y^{(i)}
$$

measures classification error.

Possible values:

| Actual | Predicted | Error |
|---|---|---|
| 1 | 1 | 0 |
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | -1 |

---

# Correct Classification

If prediction is correct:

$$
y=\hat y
$$

Then:

$$
y-\hat y=0
$$

Update becomes:

$$
\theta=\theta
$$

No change occurs.

---

# Misclassified Positive Example

Suppose:

$$
y=1,\quad \hat y=0
$$

Then:

$$
y-\hat y=1
$$

Update becomes:

$$
\theta=\theta+\alpha x
$$

---

# Intuition

When a positive point is classified incorrectly:

- the decision boundary should move toward that point,
- so the model can classify it correctly next time.

The update increases the weights in the direction of the input vector.

---

# Misclassified Negative Example

Suppose:

$$
y=0,\quad \hat y=1
$$

Then:

$$
y-\hat y=-1
$$

Update becomes:

$$
\theta=\theta-\alpha x
$$

---

# Intuition

When a negative point is classified incorrectly:

- the boundary should move away from that point.

The update decreases weights accordingly.

---

# Geometric Interpretation

The Perceptron update changes the orientation of the decision boundary.

Misclassified examples:

- push the boundary,
- rotate the separator,
- improve class separation.

---

# Why Learning Rate Matters

The learning rate:

$$
\alpha
$$

controls update size.

Small learning rate:
- slow learning.

Large learning rate:
- aggressive updates.

---

# Online Learning Behavior

Perceptron updates after each example.

This is similar to:

> Stochastic Gradient Descent.

The model continuously improves during training.

---

# Training Process

The Perceptron repeatedly:

1. selects a training example,
2. predicts output,
3. checks classification error,
4. updates weights if necessary.

This continues until classification improves.

---

# Why Perceptron Converges

If data is linearly separable:

- Perceptron eventually finds a separating boundary.

This is known as the:

> Perceptron Convergence Theorem.

---

# Relationship to Logistic Regression

The update rule resembles Logistic Regression updates.

However:

- Perceptron uses hard classification errors,
- Logistic Regression uses probabilistic gradients.

---

# Example

Suppose:

$$
x=[2,3]
$$

Prediction:

$$
\hat y=0
$$

Actual:

$$
y=1
$$

Learning rate:

$$
\alpha=0.1
$$

Update:

$$
\theta
=
\theta
+
0.1[2,3]
$$

Weights move toward the positive example.

---

# Key Insights

- Perceptron updates only when mistakes occur.
- Misclassified positive points pull the boundary closer.
- Misclassified negative points push the boundary away.
- The update rule improves classification iteratively.

---

# Common Mistakes

- Updating weights even after correct prediction
- Confusing prediction error with probability error
- Forgetting the role of learning rate
- Ignoring geometric interpretation

---

# Implementation Perspective

Perceptron learning is simple and efficient.

Although modern ML uses more advanced algorithms:

- the perceptron update idea strongly influenced neural network training methods.
