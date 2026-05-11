# Cost Function in Linear Regression

## Why Do We Need a Cost Function?

The hypothesis function makes predictions.

But how do we know whether the predictions are good or bad?

We need a way to measure prediction error.

That measurement is called the **cost function**.

---

## Goal of the Cost Function

The cost function tells us:

- how wrong the model is,
- how far predictions are from actual values.

Small cost:
- good model fit

Large cost:
- poor model fit

---

## Squared Error Cost Function

Linear Regression commonly uses Mean Squared Error (MSE).

:contentReference[oaicite:3]{index=3}

Where:

- \(J(\theta)\) = cost function
- \(m\) = number of training examples
- \(h_\theta(x^{(i)})\) = prediction
- \(y^{(i)}\) = actual value

---

## Understanding Prediction Error

Prediction error:

\[
h_\theta(x^{(i)}) - y^{(i)}
\]

If prediction equals actual value:
- error = 0

Otherwise:
- error increases.

---

## Why Squared Error?

We square errors because:

- negative and positive errors should not cancel,
- larger mistakes should be penalized more heavily.

Example:

| Error | Squared Error |
|---|---|
| 2 | 4 |
| 10 | 100 |

Large mistakes become much more expensive.

---

## Why Divide by \(m\)?

We divide by \(m\) to compute the **average error** across all training examples.

This makes the cost independent of dataset size.

---

## Why Divide by 2?

The factor:

\[
\frac{1}{2}
\]

is added for mathematical convenience.

It cancels out nicely during differentiation.

---

## Convex Cost Surface

For Linear Regression, the cost function forms a convex bowl-shaped surface.

This is extremely important.

A convex function has:

- only one global minimum,
- no local minima traps.

That means optimization becomes easier.

---

## Bowl-Shaped Intuition

Imagine placing a ball inside a bowl.

No matter where the ball starts:
- it eventually rolls toward the lowest point.

Gradient Descent behaves similarly.

---

## Relationship Between Parameters and Cost

Different parameter values produce different prediction errors.

Good parameters:
- low cost

Bad parameters:
- high cost

The objective is:

\[
\arg \min_\theta J(\theta)
\]

---

## Key Insights

- Cost function measures model error.
- Squared error punishes large mistakes.
- Lower cost means better predictions.
- Convexity guarantees one global optimum.

---

## Common Mistakes

- Forgetting to average over \(m\)
- Confusing prediction function with cost function
- Thinking cost can become negative
- Ignoring why errors are squared

---

## Implementation Perspective

In NumPy:

- predictions are computed as vectors,
- errors are computed element-wise,
- squaring and averaging are vectorized operations.

Efficient vectorization becomes critical for large datasets.
