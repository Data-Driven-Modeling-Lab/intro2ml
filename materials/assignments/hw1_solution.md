---
title: "Problem Set 1 Solution"
layout: note
category: "Assignment"
permalink: /materials/assignments/hw1_solution/
---

## Problem 3: Linear Regression and Gradient Descent

**Setup.**  
With the bias trick, each example is $x_i \in \mathbb{R}^{d+1}$ with $x_{i0}=1$, parameters $\theta \in \mathbb{R}^{d+1}$, prediction $h_\theta(x)=\theta^\top x$, and squared loss

$$
J(\theta)=\tfrac12\sum_{i=1}^n(h_\theta(x_i)-y_i)^2
=\tfrac12\|X\theta-y\|_2^2,
$$

where $X\in\mathbb{R}^{n\times(d+1)}$ has rows $x_i^\top$.

---

### (a) Bias trick $\Leftrightarrow$ hyperplane not through origin (geometry in $\mathbb{R}^2$)

In $\mathbb{R}^2$ (one real feature $x_1$ plus the constant $x_0=1$),

$$
h_\theta(x)=\theta_0\cdot 1+\theta_1 x_1=\theta_0+\theta_1 x_1,
$$

which is a line with intercept $\theta_0$. Without the constant term, $h=\theta_1 x_1$ is constrained to pass through the origin. Hence adding $x_0=1$ allows vertical shifting and the fitted line need not pass through $(0,0)$.

---

### (b) Why minimizing the raw residual is bad; absolute vs. squared loss

If we tried $L(\theta)=\sum_i (h_\theta(x_i)-y_i)$,

$$
L(\theta)=\theta^\top \!\Big(\sum_i x_i\Big)-\sum_i y_i
$$

is linear in $\theta$: if $\sum_i x_i\neq 0$, $L$ is unbounded and has no minimizer; if $\sum_i x_i=0$, *every* $\theta$ minimizes $L$.  
Absolute loss $\sum_i |h_\theta(x_i)-y_i|$ is convex but nondifferentiable at zero residuals (subgradients needed).  
Squared loss $\sum_i (h_\theta(x_i)-y_i)^2$ is convex *and* smooth, ideal for gradient-based optimization and admitting a closed-form solution.

---

### (c) Convexity and strict convexity

$$
J(\theta)=\tfrac12\theta^\top X^\top X\,\theta-\theta^\top X^\top y+\tfrac12 y^\top y,\qquad
\nabla^2 J(\theta)=X^\top X\succeq 0.
$$

Thus $J$ is convex. It is *strictly* convex iff $X$ has full column rank (so $X^\top X\succ 0$), and then the minimizer is unique.

---

### (d) Gradient and one batch GD step

Two equivalent forms:

$$
\nabla_\theta J(\theta)=\sum_{i=1}^n (h_\theta(x_i)-y_i)\,x_i
\quad\text{or}\quad
\nabla_\theta J(\theta)=X^\top(X\theta-y).
$$

Batch GD with learning rate $\alpha>0$:

$$
\boxed{\;\theta \leftarrow \theta-\alpha\,X^\top(X\theta-y)\;}
\quad\text{(equivalently }\theta\leftarrow\theta-\alpha\sum_i (h_\theta(x_i)-y_i)x_i\text{)}.
$$

---

### (e) One batch GD step on the tiny dataset

Given $(x_1,y_1)=([1,2],5)$ and $(x_2,y_2)=([1,-1],0)$, start $\theta^{(0)}=(0,0)$.

$$
r_1=\theta^{(0)\top}x_1-y_1=0-5=-5,\qquad r_2=0-0=0.
$$

$$
\nabla J(\theta^{(0)})=\sum_i r_i x_i=(-5)[1,2]+0=[-5,-10].
$$

With $\alpha=0.1$,

$$
\boxed{\theta^{(1)}=\theta^{(0)}-\alpha\nabla J(\theta^{(0)})=(0,0)-0.1[-5,-10]=(0.5,\,1.0).}
$$

---

### (f) One pass of SGD visiting $x_1$ then $x_2$ ($\alpha=0.1$)

From $\theta=(0,0)$:

$$
\text{(1) at }(x_1,y_1):\ r_1=-5,\quad
\theta\leftarrow (0,0)-0.1(-5)[1,2]=(0.5,\,1.0).
$$

$$
\text{(2) at }(x_2,y_2):\ r_2=(0.5,1.0)\!\cdot\![1,-1]-0= -0.5,\quad
\theta\leftarrow (0.5,1.0)-0.1(-0.5)[1,-1]=(0.55,\,0.95).
$$

SGD takes two cheap example-wise steps and ends at $(0.55,0.95)$; its path differs from batch GD (which landed at $(0.5,1.0)$) but both decrease $J$.

---

### (g) Level sets and step size

In $(\theta_0,\theta_1)$, $J$ is a convex quadratic; level sets are ellipses centered at the minimizer (unique if $X$ has full column rank). The gradient is normal to the level set and $-\nabla J$ points along steepest descent. If $\alpha$ is too large, steps overshoot and can oscillate/diverge (especially with elongated ellipses from poorly scaled/correlated features). With a reasonably small $\alpha$, GD moves stably toward the minimizer; rescaling/whitening accelerates convergence.

## Problem 4: Kaggle Competition

Some crucial ideas:
- Splitting the dataset into a training, validation, and test set.
- Cross validating, that is looping over a range of polynomial degrees, evaluating the training and validation loss for each, and choosing the degree that minimizes the validation loss.
- Applying the discovered model on the unseen data.

A sample solution:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X = train_df[["x1", "x2"]].values
y = train_df["y"].values

# split data into train set (to train the candidate model), val set (to evaluate the model in the cross-validation process)
# and test set (the final evaluation of the "best" model)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, shuffle=True) 

best_degree = None
best_val_mse = float("inf")
best_model = None

# cross-validation: since we are creating a polynomial feature matrix in the input, we want to check which degree to choose
# we want to avoid overfitting while minimzing the MSE.
for degree in range(1, 7):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_val_pred = model.predict(X_val_poly)
    val_mse = mean_squared_error(y_val, y_val_pred)
    
    print(f"Degree {degree} → Validation MSE: {val_mse:.4f}")
    
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_degree = degree
        best_model = (poly, model)

poly, model = best_model
X_trainval_poly = poly.fit_transform(np.vstack([X_train, X_val]))
y_trainval = np.concatenate([y_train, y_val])
model.fit(X_trainval_poly, y_trainval)

# test the best model
X_test_poly = poly.transform(X_test)
y_test_pred = model.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"\nBest degree: {best_degree}")
print(f"Test MSE: {test_mse:.4f}")

# AFTER evaluating and testing the model, we finally apply it to the unseen data
X_submit_poly = poly.transform(test_df[["x1", "x2"]].values)
y_submit = model.predict(X_submit_poly)

solution_df = pd.DataFrame({"y": y_submit})
solution_df.to_csv("solution.csv", index=False)

print("\nPredictions saved to solution.csv")
```