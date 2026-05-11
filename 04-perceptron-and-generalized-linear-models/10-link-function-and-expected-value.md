# Link Function and Expected Value

## Motivation

Generalized Linear Models (GLMs) predict outputs using probability distributions.

However:

- different distributions have different output ranges,
- raw linear models can produce unrestricted values.

We therefore need a mechanism that connects:

$$
\theta^Tx
$$

with valid expected outputs.

This mechanism is called the:

> link function.

---

# Expected Value in GLMs

In GLMs, predictions are interpreted as:

$$
\mathbb{E}[y|x]
$$

Meaning:

> expected output conditioned on input.

This probabilistic interpretation is central to GLMs.

---

# Why Expected Value Matters

Suppose:

- outputs are binary,
- outputs are counts,
- outputs are probabilities.

Directly predicting unrestricted real numbers becomes problematic.

Expected values ensure predictions remain statistically meaningful.

---

# The Linear Predictor

GLMs first compute:

$$
\eta=\theta^Tx
$$

Where:

- $\eta$ = linear predictor.

But:

$$
\eta
$$

can take any real value.

---

# Problem with Raw Linear Outputs

Suppose we predict probabilities.

Valid probabilities must satisfy:

$$
0 \le P \le 1
$$

But:

$$
\theta^Tx
$$

can produce:

$$
-100,\ 250,\ -7
$$

These are invalid probabilities.

---

# Role of the Link Function

The link function connects:

$$
\mathbb{E}[y|x]
$$

with:

$$
\eta
$$

Formally:

$$
g(\mu)=\eta
$$

Where:

$$
\mu=\mathbb{E}[y|x]
$$

---

# Inverse Link Function

Predictions are usually obtained using:

$$
\mu=g^{-1}(\eta)
$$

This converts linear predictor values into valid outputs.

---

# Logistic Regression Example

For Logistic Regression:

$$
\eta=\theta^Tx
$$

The inverse link function is sigmoid:

$$
\mu=
\frac{1}{1+e^{-\eta}}
$$

Therefore:

$$
\mathbb{E}[y|x]
=
\frac{1}{1+e^{-\theta^Tx}}
$$

---

# Why Sigmoid Works

Sigmoid converts:

$$
(-\infty,\infty)
$$

into:

$$
(0,1)
$$

making outputs valid probabilities.

---

# Linear Regression Example

For Linear Regression:

$$
\mathbb{E}[y|x]=\theta^Tx
$$

The link function is simply:

> identity.

No transformation is required.

---

# Poisson Regression Example

For count prediction:

$$
y \in \{0,1,2,\ldots\}
$$

Outputs must remain positive.

Poisson regression uses exponential link:

$$
\mu=e^\eta
$$

This guarantees positive predictions.

---

# Why Different Problems Need Different Links

Different output types impose different constraints.

| Problem Type | Valid Outputs |
|---|---|
| Regression | Any real number |
| Binary Classification | Between 0 and 1 |
| Count Prediction | Nonnegative integers |

Link functions enforce these constraints mathematically.

---

# Deep Statistical Insight

The link function transforms:

- unrestricted linear predictions,
- into statistically meaningful outputs.

This is one of the central ideas behind GLMs.

---

# Relationship to Probability Theory

The expected value:

$$
\mathbb{E}[y|x]
$$

depends on the chosen probability distribution.

The link function ensures consistency between:

- linear algebra,
- probability distributions,
- predictions.

---

# Why This is Important

Without link functions:

- predictions could become invalid,
- probabilistic interpretation would fail,
- statistical assumptions would break.

---

# Key Insights

- GLMs predict expected values.
- Link functions connect expectations with linear predictors.
- Different distributions require different link functions.
- Sigmoid is the inverse link for Logistic Regression.

---

# Common Mistakes

- Confusing link function with activation function
- Forgetting expected value interpretation
- Ignoring output constraints
- Assuming all models use sigmoid

---

# Implementation Perspective

Modern ML systems frequently use link functions internally.

Understanding them provides deeper insight into:

- probabilistic predictions,
- classification,
- statistical modeling,
- GLM theory.
