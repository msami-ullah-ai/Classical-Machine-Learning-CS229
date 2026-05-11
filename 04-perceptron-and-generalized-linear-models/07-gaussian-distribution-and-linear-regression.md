# Gaussian Distribution and Linear Regression

## Motivation

Linear Regression is often introduced algebraically.

However:

> Linear Regression also has a probabilistic interpretation.

This interpretation emerges naturally from the:

> Gaussian distribution.

Understanding this connection is extremely important because it reveals:

- why squared error appears,
- why Linear Regression works,
- how probabilistic modeling connects to optimization.

---

# Gaussian Distribution

The Gaussian distribution is also called the:

> Normal Distribution.

Its probability density function is:

$$
P(y;\mu,\sigma^2)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp
\left(
-
\frac{(y-\mu)^2}
{2\sigma^2}
\right)
$$

Where:

- $\mu$ = mean
- $\sigma^2$ = variance

---

# Gaussian Distribution in Machine Learning

Suppose outputs are continuous values:

- house prices,
- temperatures,
- stock values.

A common assumption is:

> outputs are Gaussian distributed around predictions.

---

# Linear Regression Assumption

Linear Regression assumes:

$$
y^{(i)}
=
\theta^Tx^{(i)}+\epsilon^{(i)}
$$

Where:

$$
\epsilon^{(i)}
$$

represents random noise.

---

# Noise Assumption

We assume noise follows:

$$
\epsilon^{(i)}
\sim
\mathcal{N}(0,\sigma^2)
$$

Meaning:

- errors are Gaussian distributed,
- centered around zero.

---

# Resulting Output Distribution

Because noise is Gaussian:

$$
y^{(i)}|x^{(i)}
\sim
\mathcal{N}(\theta^Tx^{(i)},\sigma^2)
$$

This means:

- predictions become Gaussian means,
- outputs fluctuate around predictions.

---

# Probabilistic Interpretation

Linear Regression predicts:

$$
\mathbb{E}[y|x]
=
\theta^Tx
$$

Meaning:

> the prediction is the expected value of the output distribution.

---

# Why Squared Error Appears

Start with Gaussian likelihood:

$$
P(y|x;\theta)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp
\left(
-
\frac{(y-\theta^Tx)^2}
{2\sigma^2}
\right)
$$

Take logarithm:

$$
\log P(y|x;\theta)
=
-\frac{(y-\theta^Tx)^2}{2\sigma^2}
+\text{constant}
$$

Maximizing likelihood becomes equivalent to minimizing:

$$
(y-\theta^Tx)^2
$$

This naturally produces:

> squared error loss.

---

# The Important Insight

Linear Regression does not randomly use squared error.

Squared error emerges naturally from:

> Gaussian probabilistic assumptions.

---

# Gaussian Distribution as Exponential Family

The Gaussian distribution also belongs to the:

> exponential family.

This connects Linear Regression to the broader GLM framework.

---

# Relationship to Logistic Regression

| Linear Regression | Logistic Regression |
|---|---|
| Gaussian outputs | Bernoulli outputs |
| Continuous values | Binary values |
| Squared error | Log-likelihood |
| Identity link | Sigmoid link |

---

# Why This Perspective Matters

This probabilistic view provides:

- statistical interpretation,
- theoretical justification,
- deeper understanding of optimization.

---

# Expected Value Interpretation

Linear Regression predicts:

$$
\mathbb{E}[y|x]
$$

the expected output value conditioned on inputs.

This interpretation becomes central in:

> Generalized Linear Models.

---

# Key Insights

- Linear Regression has a probabilistic foundation.
- Gaussian assumptions naturally produce squared error loss.
- Predictions represent expected values.
- Gaussian distribution belongs to exponential family.

---

# Common Mistakes

- Treating Linear Regression as purely algebraic
- Ignoring probabilistic assumptions
- Forgetting noise interpretation
- Confusing prediction with exact output

---

# Implementation Perspective

Most ML libraries hide these probabilistic foundations.

However:

- likelihood theory,
- Gaussian assumptions,
- statistical modeling

are deeply connected to Linear Regression optimization.
