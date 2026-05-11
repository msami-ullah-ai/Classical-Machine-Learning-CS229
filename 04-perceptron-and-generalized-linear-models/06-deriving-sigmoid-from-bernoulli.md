# Deriving Sigmoid Function from Bernoulli Distribution

## Motivation

One of the most important theoretical results in Machine Learning is:

> the sigmoid function naturally emerges from the Bernoulli distribution.

This means Logistic Regression is not an arbitrary algorithm.

Instead:
- it follows directly from probability theory.

---

# Starting with Bernoulli Distribution

Recall the Bernoulli distribution:

$$
P(y;\phi)
=
\phi^y(1-\phi)^{1-y}
$$

Where:

$$
0 \le \phi \le 1
$$

represents probability of class 1.

---

# Goal

We want to rewrite Bernoulli distribution into:

> exponential family form.

The canonical exponential family structure is:

$$
P(y;\eta)
=
b(y)
\exp
\left(
\eta T(y)-a(\eta)
\right)
$$

---

# Expanding Bernoulli Distribution

Start with:

$$
P(y;\phi)
=
\phi^y(1-\phi)^{1-y}
$$

Take logarithm inside exponential form:

$$
=
\exp
\left(
\log
\left(
\phi^y(1-\phi)^{1-y}
\right)
\right)
$$

Apply log rules:

$$
=
\exp
\left(
y\log\phi
+
(1-y)\log(1-\phi)
\right)
$$

---

# Rearranging Terms

Expand:

$$
=
\exp
\left(
y\log\phi
+
\log(1-\phi)
-
y\log(1-\phi)
\right)
$$

Group terms containing:

$$
y
$$

Result:

$$
=
\exp
\left(
y\log\frac{\phi}{1-\phi}
+
\log(1-\phi)
\right)
$$

---

# Identifying the Natural Parameter

Now compare with exponential family form.

We identify:

$$
\eta
=
\log\frac{\phi}{1-\phi}
$$

This quantity is called the:

> log-odds or logit function.

---

# Solving for $\\phi$

Now derive probability in terms of:

$$
\eta
$$

Start with:

$$
\eta
=
\log\frac{\phi}{1-\phi}
$$

Exponentiate both sides:

$$
e^\eta
=
\frac{\phi}{1-\phi}
$$

---

# Rearranging

Multiply both sides:

$$
e^\eta(1-\phi)=\phi
$$

Expand:

$$
e^\eta-e^\eta\phi=\phi
$$

Move terms involving:

$$
\phi
$$

$$
e^\eta=\phi+e^\eta\phi
$$

Factor:

$$
e^\eta=\phi(1+e^\eta)
$$

Solve for:

$$
\phi
$$

$$
\phi=
\frac{e^\eta}{1+e^\eta}
$$

---

# Final Simplification

Multiply numerator and denominator by:

$$
e^{-\eta}
$$

Result:

$$
\phi=
\frac{1}{1+e^{-\eta}}
$$

This is the:

> sigmoid function.

---

# The Amazing Result

The sigmoid function emerges naturally from:

- Bernoulli probability,
- exponential family mathematics.

This is why Logistic Regression uses sigmoid.

---

# Relationship to Logistic Regression

In Logistic Regression:

$$
\eta=\theta^Tx
$$

Substitute:

$$
\phi=
\frac{1}{1+e^{-\theta^Tx}}
$$

Therefore:

$$
h_\theta(x)
=
\frac{1}{1+e^{-\theta^Tx}}
$$

This is the Logistic Regression hypothesis.

---

# Why This is Important

This derivation shows:

> Logistic Regression is fundamentally a probabilistic model.

The sigmoid function is mathematically justified by probability theory.

---

# Understanding Log-Odds

The quantity:

$$
\log\frac{\phi}{1-\phi}
$$

measures:

> log-odds.

Odds:

$$
\frac{\phi}{1-\phi}
$$

compare probability of success against failure.

---

# Why Sigmoid is Perfect for Probability

The sigmoid function guarantees:

$$
0<\phi<1
$$

Therefore:
- outputs remain valid probabilities.

---

# Deep Insight

This derivation connects:

- probability theory,
- exponential families,
- statistical learning,
- Logistic Regression.

This is one of the most elegant derivations in Machine Learning.

---

# Key Insights

- Bernoulli distribution naturally produces sigmoid.
- Sigmoid emerges from exponential family theory.
- Logistic Regression is probabilistically grounded.
- Log-odds are central to classification theory.

---

# Common Mistakes

- Memorizing sigmoid without derivation
- Ignoring probabilistic foundations
- Confusing odds with probabilities
- Forgetting the role of the natural parameter

---

# Implementation Perspective

Modern ML libraries implement sigmoid directly.

However, understanding this derivation provides:

- deeper theoretical understanding,
- stronger intuition for probabilistic modeling,
- better grasp of Logistic Regression foundations.
