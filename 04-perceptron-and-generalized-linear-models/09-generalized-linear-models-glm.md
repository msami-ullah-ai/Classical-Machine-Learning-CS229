# Generalized Linear Models (GLMs)

## Motivation

Linear Regression works well for:

- continuous outputs.

Logistic Regression works well for:

- binary outputs.

But Machine Learning problems can involve many different types of outputs:

- counts,
- probabilities,
- positive values,
- multiclass labels.

We therefore need a more general framework.

This leads to:

> Generalized Linear Models (GLMs).

---

# Core Idea of GLMs

Generalized Linear Models extend ordinary linear models by combining:

1. a linear predictor,
2. an exponential family distribution,
3. a link function.

This framework unifies many Machine Learning algorithms.

---

# GLM Pipeline

The general GLM process is:

$$
x
\rightarrow
\theta^Tx
\rightarrow
\eta
\rightarrow
P(y|x)
\rightarrow
\mathbb{E}[y|x]
$$

This pipeline connects:

- features,
- probabilities,
- expected outputs.

---

# Step 1 — Linear Predictor

GLMs begin with a linear model:

$$
\eta=\theta^Tx
$$

Where:

- $x$ = input features
- $\theta$ = parameters
- $\eta$ = linear predictor

---

# Step 2 — Choose a Distribution

The output distribution must belong to the:

> exponential family.

Examples:

| Output Type | Distribution |
|---|---|
| Continuous | Gaussian |
| Binary | Bernoulli |
| Counts | Poisson |

---

# Step 3 — Apply Link Function

The link function connects:

$$
\mathbb{E}[y|x]
$$

with:

$$
\eta
$$

This allows different output types to be modeled appropriately.

---

# Why GLMs are Powerful

GLMs provide one unified framework for many models.

Instead of treating:

- Linear Regression,
- Logistic Regression,
- Poisson Regression

as unrelated algorithms,

GLMs show they are mathematically connected.

---

# Linear Regression as a GLM

For Linear Regression:

| Component | Choice |
|---|---|
| Distribution | Gaussian |
| Linear Predictor | $\theta^Tx$ |
| Link Function | Identity |

Result:

$$
\mathbb{E}[y|x]=\theta^Tx
$$

---

# Logistic Regression as a GLM

For Logistic Regression:

| Component | Choice |
|---|---|
| Distribution | Bernoulli |
| Linear Predictor | $\theta^Tx$ |
| Link Function | Sigmoid |

Result:

$$
P(y=1|x)
=
\frac{1}{1+e^{-\theta^Tx}}
$$

---

# Why the Linear Predictor Matters

The linear predictor:

$$
\theta^Tx
$$

captures weighted feature combinations.

This remains central across nearly all GLMs.

---

# Expected Value Interpretation

GLMs focus heavily on:

$$
\mathbb{E}[y|x]
$$

which represents:

> expected output conditioned on input.

Predictions become expected values under probabilistic models.

---

# Statistical Foundation

GLMs combine:

- probability theory,
- optimization,
- linear algebra,
- statistical learning.

This gives them strong theoretical foundations.

---

# Why Exponential Family is Required

Exponential family distributions provide:

- tractable likelihoods,
- clean derivatives,
- convenient optimization.

Without these properties:
- GLMs would become much harder to train.

---

# Examples of GLMs

| Model | Distribution |
|---|---|
| Linear Regression | Gaussian |
| Logistic Regression | Bernoulli |
| Poisson Regression | Poisson |
| Softmax Regression | Multinomial |

---

# Why GLMs Matter in ML

GLMs are foundational because they explain:

> how different ML algorithms emerge naturally from probability distributions.

This creates a unified statistical learning framework.

---

# Relationship to Deep Learning

Modern Neural Networks can be viewed as:

- highly flexible nonlinear extensions of GLMs.

Many deep learning output layers still rely on GLM principles.

---

# Key Insights

- GLMs unify many Machine Learning models.
- They combine linear predictors with probability distributions.
- Different output types require different distributions.
- Expected values play a central role in prediction.

---

# Common Mistakes

- Treating ML models as unrelated algorithms
- Ignoring probabilistic foundations
- Forgetting role of link functions
- Confusing linear predictor with final output

---

# Implementation Perspective

Modern ML frameworks frequently use GLM principles internally.

Understanding GLMs provides deeper insight into:

- classification,
- regression,
- probabilistic modeling,
- statistical learning theory.
