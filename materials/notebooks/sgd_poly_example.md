---
title: "synthetic parameters"
layout: note
category: "Jupyter Notebook"
permalink: /materials/notebooks/sgd_poly_example/
notebook_source: "sgd_poly_example.ipynb"
---

```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
num_points = 1000
var = 100
a = 1.5
b = 2

# generate data
x = np.linspace(-10, 10, num_points)
# y = 1.5 * x**2 + b + var * np.random.normal(0, 1, num_points)
y = x + 1.5 * x**3 + b + var * np.random.normal(0, 1, num_points)

# Plot
fig = plt.figure()
plt.plot(x, y, 'ro', ms=2)
plt.xlabel('x, inputs')
plt.ylabel('y, outputs')
plt.show()
```


    
![png](/materials/notebooks/sgd_poly_example/output_1_0.png)
    


```python
# Create features

def design_matrix(x, degree):
    X = np.zeros((len(x), degree+1))
    for i in range(X.shape[1]):
        X[:, i] = x**i
    return X


degree = 2
X = design_matrix(x, degree)
y = y.reshape(-1, 1)
```


```python
# Split data

n_train = int(0.8 * num_points)
n_test = num_points - n_train

shuff_index = np.random.permutation(num_points) 
X_shuffle = X[shuff_index]
y_shuffle = y[shuff_index]

X_train = X_shuffle[:n_train]
X_test = X_shuffle[n_train:]
y_train = y_shuffle[:n_train]
y_test = y_shuffle[n_train:]

# Plot training data
fig = plt.figure()
plt.plot(X_train[:, 1], y_train, 'ro', ms=2, label='Training data')
plt.plot(X_test[:, 1], y_test, 'go', ms=2, label='Test data')
plt.xlabel('x, inputs')
plt.ylabel('y, outputs')
plt.legend()
plt.show()

```


    
![png](/materials/notebooks/sgd_poly_example/output_3_0.png)
    


```python

# Fit model with gradient descent

def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((X @ theta - y)**2)

def gradient_descent(X, y, theta, learning_rate, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        theta = theta - (learning_rate/m) * X.T @ (X @ theta - y)
        J_history[i] = cost_function(X, y, theta)
    return theta, J_history

theta = np.random.randn(degree+1, 1)
learning_rate = 0.001
num_iters = 1000000
theta, J_history = gradient_descent(X_train, y_train, theta, learning_rate, num_iters)

```


```python
theta
```


    array([[ 9.47269996],
           [91.08193558],
           [-0.20010861]])


```python
# plot comparison
fig = plt.figure()
plt.plot(X_train[:, 1], y_train, 'ro', ms=2, label='Training data')
plt.plot(X_test[:, 1], y_test, 'go', ms=2, label='Test data')
plt.plot(X_train[:, 1], X_train @ theta, 'bo', ms=2, label='Model')
plt.xlabel('x, inputs')
plt.ylabel('y, outputs')
plt.legend()
plt.show()

```


    ![png](/materials/notebooks/sgd_poly_example/output_1_0.png)

```python
theta
```


    array([[ 9.47269996],
           [91.08193558],
           [-0.20010861]])


```python
# plot j history
%matplotlib widget
fig = plt.figure()
plt.plot(np.arange(num_iters)[5:], J_history[5:])
plt.xlabel('Iterations (Epochs)')
plt.ylabel('Cost function')
plt.show()

```


    ![png](/materials/notebooks/sgd_poly_example/output_2_1.png)

```python
test_loss = cost_function(X_test, y_test, theta)
train_loss = cost_function(X_train, y_train, theta)

print(f'Test loss: {test_loss}')
print(f'Train loss: {train_loss}')

```

    Test loss: 32181.613970835257
    Train loss: 29564.161369280937


```python

```
