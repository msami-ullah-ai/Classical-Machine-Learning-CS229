# Locally Weighted Linear Regression (LWLR)

Locally Weighted Linear Regression is a **non-parametric** algorithm that fits a regression model around a specific query point.

Instead of learning a global model, LWLR:
- Assigns higher weights to nearby points
- Assigns lower weights to far points

---

## 🔑 Key Idea

For a query point `x`, we compute weights using:

w(i) = exp( - (x - x_i)^2 / (2 * τ^2) )

Where:
- τ (tau) = bandwidth parameter
- Smaller τ → more local behavior
- Larger τ → more global smoothing

---

## 📌 Model Formula

θ = (Xᵀ W X)⁻¹ Xᵀ W y

Where:
- W = diagonal weight matrix
- X = feature matrix
- y = target values

---

## ⚙️ Steps

1. Choose query point `x`
2. Compute weights for all data points
3. Build diagonal matrix W
4. Solve for θ
5. Predict using θ

---

## 👍 Advantages

- Captures non-linear relationships
- Flexible model

## 👎 Disadvantages

- Computationally expensive
- Must recompute for every prediction

---

## 📊 Use Cases

- Small datasets
- Smooth curve fitting
- When global linear model fails
