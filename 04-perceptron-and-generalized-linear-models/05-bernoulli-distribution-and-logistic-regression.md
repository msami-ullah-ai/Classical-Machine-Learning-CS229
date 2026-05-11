# Bernoulli Distribution and Logistic Regression

## Motivation

Logistic Regression is fundamentally a probabilistic model.

To understand why the sigmoid function appears naturally, we first need to understand:

> the Bernoulli distribution.

This distribution models binary outcomes.

---

# Binary Random Variables

Suppose an output variable can take only two possible values:

$$
y \in \{0,1\}
$$

Examples:

| Problem | Outputs |
|---|---|
| Spam Detection | Spam / Not Spam |
| Disease Detection | Positive / Negative |
| Exam Result | Pass / Fail |

This type of random variable is modeled using the Bernoulli distribution.

---

# Bernoulli Distribution

The Bernoulli distribution defines:

$$
P(y=1)=\phi
$$

and:

$$
P(y=0)=1-\phi
$$

Where:

$$
0 \le \phi \le 1
$$

represents probability of class 1.

---

# Probability Mass Function

The Bernoulli probability distribution can be written compactly as:

$$
P(y;\phi)
=
\phi^y(1-\phi)^{1-y}
$$

This single equation handles both cases.

---

# Why This Formula Works

---

## Case 1 — When $y=1$

Substitute:

$$
y=1
$$

Result:

$$
P(y;\phi)=\phi
$$

Correct.

---

## Case 2 — When $y=0$

Substitute:

$$
y=0
$$

Result:

$$
P(y;\phi)=1-\phi
$$

Correct again.

---

# Relationship to Logistic Regression

Logistic Regression predicts:

$$
P(y=1|x)
$$

This is exactly the same type of probability modeled by the Bernoulli distribution.

Therefore:

> Logistic Regression naturally uses a Bernoulli probabilistic model.

---

# Probabilistic Interpretation

In Logistic Regression:

$$
\phi=h_\theta(x)
$$

Where:

$$
h_\theta(x)=g(\theta^Tx)
$$

Therefore:

$$
P(y=1|x;\theta)=h_\theta(x)
$$

and:

$$
P(y=0|x;\theta)=1-h_\theta(x)
$$

---

# Combined Logistic Regression Probability

Substituting into Bernoulli form:

$$
P(y|x;\theta)
=
h_\theta(x)^y
(1-h_\theta(x))^{1-y}
$$

This becomes the core probability equation used in Logistic Regression.

---

# Why Bernoulli Distribution Matters

The Bernoulli distribution provides:

- probabilistic foundation,
- likelihood formulation,
- theoretical justification for Logistic Regression.

Without it:
- Logistic Regression would just appear as an arbitrary formula.

---

# Bernoulli Distribution in Exponential Family Form

The Bernoulli distribution also belongs to the:

> exponential family.

This is extremely important because:

- Logistic Regression emerges naturally from exponential family theory.

---

# Expanding the Bernoulli Distribution

Start with:

$$
P(y;\phi)
=
\phi^y(1-\phi)^{1-y}
$$

Take logarithms inside the exponential form:

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

Using log rules:

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
y\log\frac{\phi}{1-\phi}
+
\log(1-\phi)
\right)
$$

Now it matches exponential family structure.

---

# Natural Parameter

Define:

$$
\eta
=
\log\frac{\phi}{1-\phi}
$$

This quantity is called the:

> natural parameter.

---

# Why This is Important

This derivation shows:

> Logistic Regression is not an arbitrary algorithm.

It emerges naturally from:

- Bernoulli probability,
- exponential family theory,
- statistical learning principles.

---

# Relationship to Sigmoid Function

Once we derive:

$$
\eta
=
\log\frac{\phi}{1-\phi}
$$

we can solve for:

$$
\phi
$$

This derivation produces the:

> sigmoid function.

---

# Key Insights

- Bernoulli distribution models binary outputs.
- Logistic Regression is built on Bernoulli probabilities.
- Logistic Regression naturally emerges from probability theory.
- Bernoulli belongs to the exponential family.

---

# Common Mistakes

- Treating Logistic Regression as purely algebraic
- Forgetting probabilistic interpretation
- Ignoring Bernoulli distribution foundations
- Confusing probability with class labels

---

# Implementation Perspective

In practice:

- Logistic Regression optimizes Bernoulli likelihood,
- probabilities are computed using sigmoid,
- modern classification systems heavily rely on this probabilistic foundation.
