---
title: "Sigmoid activation and its derivative"
layout: note
category: "Jupyter Notebook"
permalink: /materials/notebooks/test_file/
notebook_source: "test_file.ipynb"
---

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Forward pass
def forward_pass(x, W1, b1, W2, b2):
    # x is a column vector (d, 1)
    z1 = np.dot(W1, x) + b1  # (h, d) * (d, 1) + (h, 1) => (h, 1)
    v1 = sigmoid(z1)  # Apply activation function (h, 1)
    
    z2 = np.dot(W2, v1) + b2  # (o, h) * (h, 1) + (o, 1) => (o, 1)
    y_hat = sigmoid(z2)  # Apply activation function (o, 1)
    
    return v1, y_hat

# Backpropagation
def backpropagation(x, v1, y_hat, y, W1, b1, W2, b2, learning_rate):
    # 1. Output layer error
    error_output = y_hat - y  # Shape: (o, 1)
    d_output = error_output * sigmoid_derivative(y_hat)  # (o, 1)

    # 2. Hidden layer error (backpropagate)
    error_hidden = np.dot(W2.T, d_output)  # (h, 1)
    d_hidden = error_hidden * sigmoid_derivative(v1)  # (h, 1)

    # 3. Update weights and biases for output layer
    W2 -= learning_rate * np.dot(d_output, v1.T)  # (o, h) -= (o, 1) * (1, h)
    b2 -= learning_rate * d_output  # (o, 1)

    # 4. Update weights and biases for hidden layer
    W1 -= learning_rate * np.dot(d_hidden, x.T)  # (h, d) -= (h, 1) * (1, d)
    b1 -= learning_rate * d_hidden  # (h, 1)

    return W1, b1, W2, b2

# Initialize weights and biases
input_dim = 2  # Number of input features
hidden_dim = 2  # Number of hidden neurons
output_dim = 1  # Number of output neurons

np.random.seed(42)  # For reproducibility
W1 = np.random.rand(hidden_dim, input_dim)  # (h, d)
b1 = np.random.rand(hidden_dim, 1)  # (h, 1)
W2 = np.random.rand(output_dim, hidden_dim)  # (o, h)
b2 = np.random.rand(output_dim, 1)  # (o, 1)

# Sample input (column vector) and output
x = np.array([[0], [1]])  # Input column vector (d, 1)
y = np.array([[1]])  # Target output (o, 1)

# Learning rate
learning_rate = 0.1

# Forward pass
v1, y_hat = forward_pass(x, W1, b1, W2, b2)

# Backpropagation
W1, b1, W2, b2 = backpropagation(x, v1, y_hat, y, W1, b1, W2, b2, learning_rate)

# Display updated weights and biases
print("Updated W1:\n", W1)
print("Updated b1:\n", b1)
print("Updated W2:\n", W2)
print("Updated b2:\n", b2)
```

    Updated W1:
     [[0.37454012 0.95075702]
     [0.73199394 0.59940052]]
    Updated b1:
     [[0.15606136]
     [0.15673655]]
    Updated W2:
     [[0.06104323 0.86885486]]
    Updated b2:
     [[0.60505318]]


```python
# Training loop
epochs = 10000
for epoch in range(epochs):
    v1, y_hat = forward_pass(x, W1, b1, W2, b2)
    W1, b1, W2, b2 = backpropagation(x, v1, y_hat, y, W1, b1, W2, b2, learning_rate)

    if epoch % 1000 == 0:
        loss = np.mean((y - y_hat) ** 2)
        print(f'Epoch {epoch+1}, Loss: {loss}')
```

    Epoch 1, Loss: 0.050200055006009915
    Epoch 1001, Loss: 0.002508771745050967
    Epoch 2001, Loss: 0.0012017913077545835
    Epoch 3001, Loss: 0.0007796953977827689
    Epoch 4001, Loss: 0.0005737736312857965
    Epoch 5001, Loss: 0.0004525050473037121
    Epoch 6001, Loss: 0.000372843044332794
    Epoch 7001, Loss: 0.0003166239447419777
    Epoch 8001, Loss: 0.0002748846634192015
    Epoch 9001, Loss: 0.00024270155003633162

