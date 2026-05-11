# Newton’s Method

## Motivation

Gradient Descent is a first-order optimization algorithm.

It uses:

- gradients,
- slopes,
- first derivatives.

Although effective, Gradient Descent may require:

- many iterations,
- small learning rates,
- slow convergence.

Newton’s Method is a more advanced optimization technique that uses:

> second-order information.

---

# Core Idea of Newton’s Method

Newton’s Method improves optimization by considering:

- slope,
- curvature of the function.

Instead of taking small gradual steps:

- it makes smarter jumps toward the optimum.

This often leads to much faster convergence.

---

# Intuition Behind Newton’s Method

Suppose we want to find where:

$$
f(\theta)=0
$$

Newton’s Method:

1. starts with an initial guess,
2. draws tangent line at current point,
3. follows tangent to the x-axis,
4. uses intersection as the next estimate.

This process repeats iteratively.

---

# Newton’s Method Update Rule

For a single variable:

$$
\theta^{(t+1)}
=
\theta^{(t)}
-
\frac{f(\theta^{(t)})}
{f'(\theta^{(t)})}
$$

Where:

- $f(\theta)$ = function value
- $f'(\theta)$ = derivative/slope

---

# Understanding the Formula

The numerator:

$$
f(\theta^{(t)})
$$

measures how far we are from the solution.

The denominator:

$$
f'(\theta^{(t)})
$$

measures the slope at the current point.

Together:

- they determine how large the next jump should be.

---

# Why Newton’s Method Converges Faster

Gradient Descent uses only slope information.

Newton’s Method uses:

- slope,
- curvature.

Because it understands the shape of the function better:

- updates become much smarter,
- convergence becomes significantly faster.

---

# Quadratic Convergence

Newton’s Method often exhibits:

> quadratic convergence.

This means:

- error decreases extremely rapidly near the optimum.

Once close to the solution:

- convergence becomes dramatically faster than Gradient Descent.

---

# Newton’s Method for Optimization

For optimization problems:

we want:

$$
f'(\theta)=0
$$

because minima and maxima occur where slope equals zero.

Applying Newton’s Method:

$$
\theta^{(t+1)}
=
\theta^{(t)}
-
\frac{f'(\theta^{(t)})}
{f''(\theta^{(t)})}
$$

Now:

- numerator = first derivative,
- denominator = second derivative.

---

# Why Second Derivative Matters

The second derivative measures:

> curvature.

It tells us:

- how sharply the function bends,
- whether curvature is steep or flat.

This allows Newton’s Method to adapt its step sizes intelligently.

---

# Relationship to Logistic Regression

In Logistic Regression:

- we maximize log-likelihood,
- Newton’s Method can optimize parameters efficiently.

Compared to Gradient Descent:

- fewer iterations are often needed.

---

# Visual Intuition

Imagine standing on a curved surface.

Gradient Descent:
- only checks slope.

Newton’s Method:
- checks both slope and curvature.

Because it understands the terrain better:
- it can jump much closer to the optimum.

---

# Advantages of Newton’s Method

## Faster Convergence

Usually requires fewer iterations.

---

## Smarter Updates

Uses curvature information.

---

## Powerful for Convex Optimization

Works especially well for convex problems like Logistic Regression.

---

# Disadvantages of Newton’s Method

## Computationally Expensive

Requires second derivatives.

---

## Hessian Matrix Computation

For multivariable problems:

- second derivatives become expensive,
- matrix inversion may be costly.

---

## Memory Intensive

Storing second-order information can become expensive for large models.

---

# Comparison with Gradient Descent

| Gradient Descent | Newton’s Method |
|---|---|
| Uses first derivatives | Uses first + second derivatives |
| Slower convergence | Faster convergence |
| Cheap iterations | Expensive iterations |
| Simpler | More mathematically complex |

---

# When Newton’s Method Works Well

Newton’s Method performs well when:

- feature count is moderate,
- second derivatives are computable,
- optimization surface is smooth.

---

# Why Newton’s Method is Important

Newton’s Method is foundational in optimization theory.

Many advanced optimization algorithms are inspired by:

- second-order methods,
- curvature-aware optimization.

---

# Key Insights

- Newton’s Method uses curvature information.
- Second derivatives improve optimization speed.
- Convergence is often much faster than Gradient Descent.
- Each iteration is computationally more expensive.

---

# Common Mistakes

- Confusing first and second derivatives
- Assuming Newton’s Method is always better
- Ignoring computational cost
- Forgetting curvature interpretation

---

# Implementation Perspective

In practice:

- Newton’s Method is powerful for smaller optimization problems,
- deep learning usually avoids full Newton updates because Hessian computation is expensive,
- many modern optimizers approximate second-order behavior.
