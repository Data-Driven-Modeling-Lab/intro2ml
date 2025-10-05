---
title: "Problem 4: Kaggle Competition"
layout: note
category: "Assignment"
permalink: /materials/assignments/hw1_solution/
---

## Problem 4: Kaggle Competition

Some crucial ideas:
- Splitting the dataset into a training, validation, and test set.
- Cross validating, that is looping over a range of polynomial degrees, evaluating the training and validation loss for each, and choosing the degree that minimizes the validation loss.
- Applying the discovered model on the unseen data.

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