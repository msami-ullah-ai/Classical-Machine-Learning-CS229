# Logistic Regression Explained

Logistic Regression is used for **binary classification** problems.

---

## 🔑 Key Idea

Instead of predicting a value, it predicts a **probability**:

p = 1 / (1 + e^(-z))

Where:
z = θᵀx

---

## 📌 Sigmoid Function

Transforms output into range (0,1):

σ(z) = 1 / (1 + e^(-z))

---

## 🎯 Decision Boundary

- If p ≥ 0.5 → class = 1
- Else → class = 0

---

## 📉 Cost Function

J(θ) = - (1/m) Σ [ y log(h(x)) + (1-y) log(1-h(x)) ]

---

## ⚙️ Gradient Descent Update

θ = θ - α * ∂J/∂θ

---

## 👍 Advantages

- Simple and fast
- Works well for linearly separable data
- Outputs probabilities

## 👎 Disadvantages

- Cannot capture complex non-linear patterns (without feature engineering)

---

## 📊 Applications

- Spam detection
- Disease prediction
- Binary classification problems
