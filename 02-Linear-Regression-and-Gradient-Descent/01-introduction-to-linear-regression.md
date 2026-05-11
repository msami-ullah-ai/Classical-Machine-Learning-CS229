# Introduction to Linear Regression
- if house size increases,
- predicted house price should also increase.

The parameters $\theta_0$ and $\theta_1$ control:

- where the line starts,
- how steep the line is.

---

## Training Examples

Each row in the dataset is called a training example.

Notation:

$$
(x^{(i)}, y^{(i)})
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

## Goal of Linear Regression

The goal is to:

- learn relationships from data,
- make accurate predictions,
- minimize prediction error.
