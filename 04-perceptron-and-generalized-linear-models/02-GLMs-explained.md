# Generalized Linear Models (GLMs) Explained

## 1. What are GLMs?

Generalized Linear Models (GLMs) extend linear regression to handle:
- Different types of output variables
- Non-linear relationships via link functions

The general form is:

$$
g(\mathbb{E}[y \mid x]) = w \cdot x + b
$$

Where:
- $g(\cdot)$ is the **link function**
- $\mathbb{E}[y \mid x]$ is the expected value of $y$ given $x$
- $w \cdot x + b$ is the **linear predictor**

---

## 2. Key Components

A GLM consists of three main parts:

### 1. Linear Predictor

$$
\eta = w \cdot x + b
$$

This is the same as in standard linear regression.

---

### 2. Link Function

The link function connects the expected value of the output to the linear predictor:

$$
g(\mu) = \eta
$$

Where:
- $\mu = \mathbb{E}[y \mid x]$

---

### 3. Output Distribution

The output $y$ is assumed to come from a distribution in the **exponential family**, such as:
- Gaussian (for regression)
- Bernoulli (for binary classification)
- Poisson (for count data)

---

## 3. Common Examples of GLMs

### Linear Regression

- Link function: Identity  
- Distribution: Gaussian  

$$
y = w \cdot x + b
$$

---

### Logistic Regression

- Link function: Sigmoid  
- Distribution: Bernoulli  

$$
P(y = 1 \mid x) = \frac{1}{1 + e^{-(w \cdot x + b)}}
$$

---

### Poisson Regression

- Link function: Log  
- Distribution: Poisson  

$$
\log(\lambda) = w \cdot x + b
$$

---

## 4. Exponential Family (Important Concept)

GLMs assume the output distribution belongs to the exponential family:

$$
p(y \mid \theta) = h(y)\exp\big(\theta^T T(y) - A(\theta)\big)
$$

Where:
- $\theta$ = natural parameter  
- $T(y)$ = sufficient statistics  
- $A(\theta)$ = log-partition function  

---

## 5. Why GLMs Matter

GLMs provide:
- A unified framework for many models
- Probabilistic interpretation
- Flexibility for different data types

They generalize:
- Linear regression → continuous outputs  
- Logistic regression → classification  
- Poisson regression → counts  

---

## 6. Training GLMs

GLMs are typically trained using **Maximum Likelihood Estimation (MLE)**.

For example, logistic regression minimizes the loss:

$$
\ell(w) = \sum_i \log\big(1 + e^{-y_i (w \cdot x_i)}\big)
$$

Optimization methods:
- Gradient Descent
- Stochastic Gradient Descent (SGD)

---

## 7. Intuition

Think of GLMs as a 3-step pipeline:

1. Compute a linear score:  
   $w \cdot x + b$

2. Apply link function  

3. Get meaningful output:
   - Probability (logistic)
   - Real value (linear regression)
   - Count (Poisson)

---

## 8. GLMs vs Perceptron

| Feature        | Perceptron        | GLMs                  |
|----------------|------------------|----------------------|
| Output         | Hard labels       | Probabilities        |
| Learning       | Update rule       | Likelihood-based     |
| Interpretation | No                | Yes                  |
| Flexibility    | Low               | High                 |

---

## 9. Key Takeaways

- GLMs extend linear models to broader settings  
- They rely on:
  - Linear predictor
  - Link function
  - Output distribution  
- Logistic regression is the most widely used GLM  

---

## 10. Summary

Generalized Linear Models:
- Are powerful and flexible  
- Provide probabilistic outputs  
- Unify multiple important ML models under one framework  
