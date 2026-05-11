# Cost Function

## Why Do We Need a Cost Function?

The hypothesis function makes predictions.

But predictions alone are not enough.

We need a way to measure:

- how wrong the model is,
- how far predictions are from actual values,
- whether the model is improving during training.

That measurement is called the **cost function**.

---

## Goal of the Cost Function

The cost function evaluates model performance.

- small cost = good predictions
- large cost = poor predictions

The learning algorithm tries to minimize this cost.

---

## Squared Error Cost Function

Linear Regression commonly uses **Mean Squared Error (MSE)**.

$$
J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2
$$

Where:

- $J(\theta)$ = cost function
- $m$ = number of training examples
- $h_\theta(x^{(i)})$ = predicted value
- $y^{(i)}$ = actual value

---

## Understanding Prediction Error

Prediction error:

$$
h_\theta(x^{(i)}) - y^{(i)}
$$

This tells us how far the prediction is from the true value.

### If prediction equals actual value:

$$
h_\theta(x^{(i)}) = y^{(i)}
$$

then:

$$
\text{error} = 0
$$

which means perfect prediction.

---

## Why Do We Square Errors?

We square errors because:

- positive and negative errors should not cancel each other,
- large mistakes should be penalized more heavily.

Example:

| Error | Squared Error |
|---|---|
| 2 | 4 |
| 10 | 100 |

A larger error becomes significantly more expensive.

This forces the model to avoid large mistakes.

---

## Why Divide by $m$?

We divide by $m$ to compute the **average error** across all training examples.

Without averaging:

- larger datasets would automatically produce larger costs.

Averaging makes the cost independent of dataset size.

---

## Why Divide by 2?

The factor:

$$
\frac{1}{2}
$$

is added for mathematical convenience.

During differentiation:

- the exponent 2 cancels with $\frac{1}{2}$,
- making equations cleaner.

---

## Relationship Between Parameters and Cost

Different parameter values produce different prediction errors.

Good parameters:
- low cost

Bad parameters:
- high cost

The objective of training is:

$$
\arg\min_{\theta} J(\theta)
$$

which means:

> find parameter values that minimize the cost function.

---

## Convex Cost Surface

For Linear Regression, the cost function forms a convex bowl-shaped surface.

This is extremely important.

A convex function has:

- only one global minimum,
- no local minima traps.

That means optimization becomes easier and more reliable.

---

## Bowl-Shaped Intuition

Imagine placing a ball inside a bowl.

No matter where the ball starts:

- it eventually rolls toward the lowest point.

Gradient Descent behaves similarly while minimizing the cost function.

---

## Contour Representation

Contour lines connect points having the same cost.

- outer contours = higher error
- inner contours = lower error

The center represents the minimum cost.

Gradient Descent moves from outer contours toward the center.

---

## Cost Function and Model Quality

The cost function acts like a score.

### Large Cost

- predictions are far from actual values,
- model performs poorly.

### Small Cost

- predictions are close to actual values,
- model fits data well.

---

## Importance of the Cost Function

Without the cost function:

- the model would not know whether it is improving,
- optimization would be impossible.

The cost function provides the learning signal used during training.

---

## Key Insights

- Cost function measures prediction error.
- Squared error penalizes large mistakes heavily.
- Lower cost means better model performance.
- Convexity guarantees one global optimum.

---

## Common Mistakes

- Forgetting to average over $m$
- Confusing hypothesis function with cost function
- Thinking cost can become negative
- Ignoring why errors are squared
- Assuming lower training cost always means better generalization

---

## Implementation Perspective

In practice:

- predictions are computed as vectors,
- errors are computed element-wise,
- squaring and averaging are vectorized operations.

Using NumPy, the entire cost function can be computed efficiently without loops.
