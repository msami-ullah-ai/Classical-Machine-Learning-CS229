# Sigmoid Function

## Why We Need the Sigmoid Function

Logistic Regression predicts probabilities.

Probabilities must always remain between:

$$
0 \leq P \leq 1
$$

However:

$$
\theta^Tx
$$

can produce any real number:

$$
(-\infty,\infty)
$$

We therefore need a function that converts any real value into a valid probability.

This function is called the:

> Sigmoid Function.

---

# Definition of the Sigmoid Function

The sigmoid function is defined as:

$$
g(z)=\frac{1}{1+e^{-z}}
$$

Where:

- $z$ can be any real number,
- output always lies between 0 and 1.

---

# Logistic Regression Hypothesis

In Logistic Regression:

$$
z=\theta^Tx
$$

Therefore:

$$
h_\theta(x)=g(\theta^Tx)
$$

Expanded form:

$$
h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}
$$

---

# Understanding the Shape

The sigmoid function produces an:

> S-shaped curve.

Characteristics:

- very negative values → output near 0
- very positive values → output near 1
- values near 0 → output near 0.5

---

# Important Output Range

The sigmoid function always satisfies:

$$
0<g(z)<1
$$

This makes it perfect for probability prediction.

---

# Behavior for Large Positive Values

Suppose:

$$
z \gg 0
$$

Then:

$$
e^{-z}\approx0
$$

Result:

$$
g(z)\approx1
$$

The model becomes highly confident about class 1.

---

# Behavior for Large Negative Values

Suppose:

$$
z \ll 0
$$

Then:

$$
e^{-z}\rightarrow\infty
$$

Result:

$$
g(z)\approx0
$$

The model becomes highly confident about class 0.

---

# Behavior at $z=0$

Suppose:

$$
z=0
$$

Then:

$$
g(0)=\frac{1}{2}=0.5
$$

This represents maximum uncertainty.

---

# Probability Interpretation

The sigmoid output can be interpreted as:

$$
P(y=1|x;\theta)
$$

Example:

$$
h_\theta(x)=0.92
$$

means:

> 92% probability that input belongs to class 1.

---

# Why the Sigmoid Function is Smooth

The sigmoid curve is:

- continuous,
- differentiable,
- smooth everywhere.

This is extremely important because:

- optimization algorithms require derivatives.

---

# Relationship to Decision Boundary

The decision boundary occurs when:

$$
h_\theta(x)=0.5
$$

Since:

$$
g(0)=0.5
$$

the boundary becomes:

$$
\theta^Tx=0
$$

---

# Why Sigmoid is Important in ML

The sigmoid function appears throughout Machine Learning:

- Logistic Regression
- Neural Networks
- Probabilistic models
- Deep Learning

It is one of the foundational activation functions.

---

# Intuition Behind the Transformation

The sigmoid function acts like:

> a probability converter.

It takes:

$$
(-\infty,\infty)
$$

and compresses everything into:

$$
(0,1)
$$

---

# Key Insights

- Sigmoid converts real numbers into probabilities.
- Outputs always remain between 0 and 1.
- Logistic Regression uses sigmoid for classification.
- The sigmoid curve is smooth and differentiable.

---

# Common Mistakes

- Forgetting sigmoid output is probabilistic
- Confusing linear score with probability
- Assuming sigmoid outputs exact class labels
- Ignoring why differentiability matters

---

# Implementation Perspective

In NumPy:

```python
sigmoid = 1 / (1 + np.exp(-z))
```

Sigmoid computation is fully vectorized in modern ML systems.
