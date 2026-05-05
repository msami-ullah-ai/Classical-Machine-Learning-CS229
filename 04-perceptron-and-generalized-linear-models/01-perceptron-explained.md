# Perceptron Explained

## 1. What is a Perceptron?

The perceptron is one of the simplest binary classification algorithms. It is a type of linear classifier that separates data using a straight line (or hyperplane in higher dimensions).

It predicts labels using:

\[
y = \text{sign}(w \cdot x + b)
\]

Where:
- \( w \) = weight vector  
- \( x \) = input features  
- \( b \) = bias  
- \( y \in \{-1, +1\} \)

---

## 2. Intuition

The perceptron tries to find a line (or plane) that separates two classes.

- If the data is **linearly separable**, perceptron will converge.
- If not, it may **never converge**.

---

## 3. Algorithm

Initialize:
- \( w = 0 \), \( b = 0 \)

Repeat until convergence:
1. Loop through each training example \( (x_i, y_i) \)
2. Predict:
   \[
   \hat{y}_i = \text{sign}(w \cdot x_i + b)
   \]
3. If misclassified:
   \[
   w = w + y_i x_i
   \]
   \[
   b = b + y_i
   \]

---

## 4. Update Rule Intuition

- If prediction is wrong:
  - Move weights **toward correct direction**
- If correct:
  - Do nothing

---

## 5. Geometric Interpretation

- Decision boundary:  
  \[
  w \cdot x + b = 0
  \]
- This defines a line (2D) or plane (higher dimensions)
- The perceptron adjusts this boundary to separate data

---

## 6. Convergence Theorem

If data is linearly separable, the perceptron will:
- Converge in a finite number of steps
- Find a separating hyperplane

---

## 7. Limitations

- Only works for **linearly separable data**
- Cannot solve problems like XOR
- No probabilistic interpretation

---

## 8. Key Takeaways

- Simple and foundational algorithm
- Introduces idea of linear decision boundaries
- Basis for more advanced models (SVMs, neural networks)

---

## 9. Pseudocode

