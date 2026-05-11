# Learning Rate and Convergence

## What is the Learning Rate?

The learning rate controls how large a step Gradient Descent takes during parameter updates.

It is represented by:

$$
\alpha
$$

in the update rule:

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta)
$$

---

## Intuition Behind the Learning Rate

The derivative tells us:

- which direction to move,
- how steep the slope is.

The learning rate determines:

- how far we move in that direction.

Think of it as the step size used while moving downhill.

---

# Small Learning Rate

## What Happens?

If the learning rate is very small:

$$
\alpha \ll 1
$$

then parameter updates become tiny.

Example:

- very slow movement toward minimum,
- training may take thousands of iterations.

---

## Advantages of Small Learning Rate

- stable updates,
- smooth convergence,
- less risk of overshooting.

---

## Disadvantages of Small Learning Rate

- training becomes extremely slow,
- optimization may take too long,
- computational cost increases.

---

# Large Learning Rate

## What Happens?

If the learning rate is too large:

$$
\alpha \gg 1
$$

updates become very aggressive.

Instead of gradually approaching the minimum:

- parameters jump around,
- updates become unstable,
- optimization may fail completely.

---

## Overshooting Problem

Suppose the minimum lies here:

```text
      \\
       \\
        \\
         O
        / \\
       /   \\
```

A very large learning rate causes the algorithm to jump across the minimum repeatedly.

Instead of converging:

- it oscillates,
- cost may even increase.

---

## Divergence

If updates become too large:

- the cost function keeps increasing,
- parameters explode,
- Gradient Descent diverges.

This means optimization completely fails.

---

# Choosing the Learning Rate

Selecting a good learning rate is extremely important.

Common practice:

Try multiple values such as:

$$
0.1,\ 0.01,\ 0.001
$$

and observe convergence behavior.

---

## Good Learning Rate Characteristics

A good learning rate should:

- reduce cost steadily,
- converge reasonably fast,
- avoid instability.

---

# Convergence

## What is Convergence?

Gradient Descent converges when:

- parameter updates become very small,
- cost stops decreasing significantly.

At convergence:

- parameters stabilize,
- the model reaches near-optimal values.

---

## Visual Intuition of Convergence

Imagine rolling a ball inside a bowl.

Initially:
- movement is fast,
- slope is steep.

Near the bottom:
- slope becomes smaller,
- movement slows down,
- the ball eventually settles.

Gradient Descent behaves similarly.

---

# Why Updates Become Smaller Near Minimum

Near the minimum:

$$
\frac{\partial}{\partial \theta_j}J(\theta) \approx 0
$$

This means:

- slope becomes nearly flat,
- parameter updates naturally become smaller.

Even with a fixed learning rate:

- Gradient Descent slows down automatically near the optimum.

---

# Monitoring Convergence

We usually monitor:

$$
J(\theta)
$$

during training.

### Healthy Convergence

- cost decreases smoothly over iterations.

### Problematic Behavior

- cost oscillates,
- cost increases,
- cost becomes unstable.

These often indicate a bad learning rate.

---

# Stopping Criteria

Training can stop when:

- cost improvement becomes negligible,
- maximum iterations are reached,
- gradients become extremely small.

---

# Epochs

An epoch means:

> one complete pass through the training dataset.

Example:

If a dataset contains 1000 examples:

- processing all 1000 examples once = 1 epoch.

---

# Relationship Between Epochs and Convergence

More epochs generally allow:

- better learning,
- lower cost,
- improved parameter estimates.

However:

- too many epochs may waste computation,
- or lead to overfitting in some models.

---

# Why Learning Rate Matters So Much

The learning rate directly affects:

- training speed,
- convergence stability,
- optimization quality.

A poor learning rate can ruin training completely.

---

# Practical Learning Rate Strategy

Typical workflow:

1. start with moderate learning rate,
2. observe cost behavior,
3. adjust if necessary.

If cost decreases too slowly:
- increase learning rate.

If cost oscillates:
- decrease learning rate.

---

# Key Insights

- Learning rate controls update size.
- Small learning rate causes slow convergence.
- Large learning rate causes instability.
- Gradient Descent converges near the minimum.
- Monitoring cost helps detect training problems.

---

# Common Mistakes

- Using excessively large learning rates
- Assuming larger learning rate always means faster training
- Ignoring divergence signs
- Stopping training too early
- Confusing epochs with iterations

---

# Implementation Perspective

In practice:

- learning rate is one of the most important hyperparameters,
- modern ML systems often use adaptive learning rates,
- convergence is monitored automatically during training.

Choosing the right learning rate can dramatically improve optimization performance.
