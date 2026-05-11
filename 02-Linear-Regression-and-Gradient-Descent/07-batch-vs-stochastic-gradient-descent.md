# Batch Gradient Descent vs Stochastic Gradient Descent

## Why Different Gradient Descent Variants Exist

Gradient Descent computes parameter updates using training data.

However, datasets can become extremely large.

Using the entire dataset for every update may become computationally expensive.

This leads to different variants of Gradient Descent.

The two most important are:

- Batch Gradient Descent (BGD)
- Stochastic Gradient Descent (SGD)

---

# Batch Gradient Descent

## Definition

Batch Gradient Descent uses the **entire training dataset** before updating parameters.

Update rule:

$$
\theta_j :=
\theta_j
-
\alpha
\frac{1}{m}
\sum_{i=1}^{m}
(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
$$

---

## How Batch Gradient Descent Works

Steps:

1. compute predictions for all training examples,
2. compute total cost,
3. compute gradients using entire dataset,
4. update parameters once.

---

## Characteristics of Batch GD

### Advantages

- stable updates,
- smooth convergence,
- accurate gradient estimates.

### Disadvantages

- computationally expensive,
- slow for huge datasets,
- requires processing all examples before updating.

---

## Intuition Behind Batch GD

Batch GD waits until it sees the entire dataset before making updates.

It behaves like:

> carefully analyzing all evidence before making a decision.

---

# Stochastic Gradient Descent (SGD)

## Definition

Stochastic Gradient Descent updates parameters using **one training example at a time**.

Update rule:

$$
\theta_j :=
\theta_j
-
\alpha
(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
$$

---

## How SGD Works

For each training example:

1. compute prediction,
2. compute error,
3. update parameters immediately,
4. move to next example.

---

## Characteristics of SGD

### Advantages

- much faster updates,
- efficient for large datasets,
- starts learning immediately.

### Disadvantages

- noisy updates,
- unstable convergence,
- fluctuates near minimum.

---

# Why SGD is Noisy

SGD only sees one example at a time.

That means:

- updates are based on incomplete information,
- gradients vary between examples.

As a result:

- optimization path becomes noisy,
- parameter updates fluctuate.

---

# Visual Intuition

## Batch GD

Smooth path toward minimum.

```text
Smooth and steady
```

---

## SGD

Zig-zag noisy movement.

```text
Noisy but faster
```

---

# Why SGD Can Still Work Well

Even though SGD is noisy:

- its updates are frequent,
- it often reaches good solutions faster.

For very large datasets:

- SGD is usually preferred.

---

# Comparing Batch GD and SGD

| Batch Gradient Descent | Stochastic Gradient Descent |
|---|---|
| Uses entire dataset | Uses one example at a time |
| Stable updates | Noisy updates |
| Smooth convergence | Fluctuating convergence |
| Computationally expensive | Computationally efficient |
| Slower updates | Faster updates |
| Better for small datasets | Better for large datasets |

---

# Computational Cost

Suppose dataset contains:

$$
m = 1,000,000
$$

examples.

Batch GD:
- processes all million examples before every update.

SGD:
- updates immediately after one example.

This makes SGD dramatically faster for huge datasets.

---

# Convergence Behavior

## Batch GD

- converges smoothly,
- follows stable path,
- approaches minimum steadily.

---

## SGD

- fluctuates around minimum,
- may never settle perfectly,
- but often reaches good solutions quickly.

---

# Epochs in SGD

An epoch means:

> processing all training examples once.

In SGD:

- many parameter updates happen within one epoch,
- because updates occur after every example.

---

# Mini-Batch Gradient Descent

Modern deep learning often uses:

> Mini-Batch Gradient Descent

Instead of:

- all examples,
- or one example,

it uses small batches such as:

$$
32,\ 64,\ 128
$$

examples at a time.

This balances:

- stability,
- speed,
- computational efficiency.

---

# When to Use Batch GD

Use Batch GD when:

- dataset is small,
- memory is sufficient,
- stable convergence is important.

---

# When to Use SGD

Use SGD when:

- dataset is very large,
- faster updates are needed,
- computational efficiency matters.

---

# Key Insights

- Batch GD uses all training examples.
- SGD updates after every example.
- Batch GD is stable but slower.
- SGD is faster but noisy.
- Large datasets often favor SGD.

---

# Common Mistakes

- Thinking SGD is always better
- Ignoring noisy convergence
- Confusing epochs with updates
- Assuming Batch GD scales well to huge datasets

---

# Implementation Perspective

In practice:

- modern ML frameworks use vectorized mini-batches,
- GPU acceleration works efficiently with mini-batch training,
- SGD variants dominate deep learning optimization today.
