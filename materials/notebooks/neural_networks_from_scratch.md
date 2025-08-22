---
title: "Initialize weights and biases"
layout: note
category: "Jupyter Notebook"
permalink: /materials/notebooks/neural_networks_from_scratch/
notebook_source: "neural_networks_from_scratch.ipynb"
---

```python
import numpy as np
```

### Step 2: Define the Activation Function


```python

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)  # Derivative of the sigmoid function

```

### Step 3: Initialize Weights and Biases


```python
# Initialize weights and biases
np.random.seed(42)
input_dim = 2  # Number of input features
hidden_dim = 2  # Number of hidden neurons
output_dim = 1  # Number of output neurons

# Weights and biases as column vectors
W1 = np.random.rand(hidden_dim, input_dim)  # Shape: (h, d)
b1 = np.random.rand(hidden_dim, 1)  # Shape: (h, 1)
W2 = np.random.rand(output_dim, hidden_dim)  # Shape: (o, h)
b2 = np.random.rand(output_dim, 1)  # Shape: (o, 1)


print("Initial Weights (Input to Hidden):\n", W1)
print("Initial Weights (Hidden to Output):\n", W2)

```

    Initial Weights (Input to Hidden):
     [[0.37454012 0.95071431]
     [0.73199394 0.59865848]]
    Initial Weights (Hidden to Output):
     [[0.05808361 0.86617615]]


### Step 4: Forward Pass

For a single example, the value of the hidden layer after applying the activation function, $ v^{[1]} $, is computed as follows:

$$
z^{[1]} = W^{[1]} x + b^{[1]}
$$
$$
v^{[1]} = \sigma(z^{[1]})
$$

Where:
- $ x $ is the input vector (shape: $d \times 1$).
- $ W^{[1]} $ is the weight matrix between the input and hidden layer (shape: $h \times d$).
- $ b^{[1]} $ is the bias vector for the hidden layer (shape: $h \times 1$) (broadcasted over samples when batched).

Similarly, for the output layer:

$$
z^{[2]} = W^{[2]}v^{[1]}  + b^{[2]}
$$
$$
\hat{y} = \sigma(z^{[2]})
$$

Where:
- $ W^{[2]} $ is the weight matrix between the hidden and output layer (shape: $o \times h$).
- $ b^{[2]} $ is the bias for the output layer (shape: $o \times 1$).


```python

# Forward pass
def forward_pass(x, W1, b1, W2, b2):
    # x is a column vector (d, 1)
    z1 = np.dot(W1, x) + b1  # (h, d) * (d, 1) + (h, 1) => (h, 1)
    v1 = sigmoid(z1)  # Apply activation function (h, 1)

    z2 = np.dot(W2, v1) + b2  # (o, h) * (h, 1) + (o, 1) => (o, 1)
    y_hat = sigmoid(z2)  # Apply activation function (o, 1)
    
    return v1, y_hat

```


### Step 5: Loss Calculation

The loss function measures the difference between the actual and predicted outputs. For this implementation, we use the Mean Squared Error (MSE) as the loss function:

$$
Loss = \frac{1}{n} \sum_{i=1}^{n} \left(y^{(i)} - \hat{y}^{(i)} \right)^2
$$


```python
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Sample input and output for testing
x_sample = np.array([[0], [1]])  # Example input
y_true_sample = np.array([[1]])  # Example true output

# Perform forward pass
v1_sample, y_pred_sample = forward_pass(x_sample, W1, b1, W2, b2)

# Compute loss
loss_sample = compute_loss(y_true_sample, y_pred_sample)
print(f'Loss for the initialized weights: {loss_sample}')
```

    Loss for the initialized weights: 0.05084754666230897


### Step 6: Backpropagation

#### 1. Output Layer Gradients:
We first compute the error at the output layer:

$$
\delta^{[2]} = \frac{\partial L}{\partial z^{[2]}} = (y - \hat{y}) \cdot \sigma'(z^{[2]})
$$
Where:
- $ \delta^{[2]} $ is the error at the output layer.
- $ \sigma'(z^{[2]}) = \hat{y} \cdot (1 - \hat{y}) $ is the derivative of the sigmoid function.

The gradients for the weights and biases between the hidden and output layers are:

$$
\frac{\partial L}{\partial W^{[2]}} = v^{[1]T} \delta^{[2]}
$$
$$
\frac{\partial L}{\partial b^{[2]}} = \delta^{[2]}
$$

#### 2. Hidden Layer Gradients:
Next, we compute the error at the hidden layer:

$$
\delta^{[1]} = \delta^{[2]} W^{[2]T} \cdot \sigma'(z^{[1]})
$$
Where:
- $ \sigma'(z^{[1]}) = v^{[1]} \cdot (1 - v^{[1]}) $ is the derivative of the sigmoid function at the hidden layer.

The gradients for the weights and biases between the input and hidden layers are:

$$
\frac{\partial L}{\partial W^{[1]}} = X^T \delta^{[1]}
$$
$$
\frac{\partial L}{\partial b^{[1]}} = \delta^{[1]}
$$


```python
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

```


```python

# Sample data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])  # XOR problem

# # Sample input (column vector) and output
# inputs = np.array([[0], [1]])  # Input column vector (d, 1)
# outputs = np.array([[1]])  # Target output (o, 1)


# Training parameters
epochs = 10000
learning_rate = 0.1

# Training loop
epochs = 10000
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(inputs)):
        x = inputs[i].reshape(-1, 1)
        y = outputs[i].reshape(-1, 1)

        v1, y_hat = forward_pass(x, W1, b1, W2, b2) # intermediate results (such as v1) are saved for backpropagation
        W1, b1, W2, b2 = backpropagation(x, v1, y_hat, y, W1, b1, W2, b2, learning_rate)

        total_loss += compute_loss(y, y_hat)

    if epoch % 1000 == 0:
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(inputs)}')

```

    Epoch 1, Loss: 0.3239537158130419
    Epoch 1001, Loss: 0.24817211793449434
    Epoch 2001, Loss: 0.21459076862934467
    Epoch 3001, Loss: 0.15936526038440854
    Epoch 4001, Loss: 0.04888994977531156
    Epoch 5001, Loss: 0.016090123221200778
    Epoch 6001, Loss: 0.00861826270292999
    Epoch 7001, Loss: 0.005697165950060215
    Epoch 8001, Loss: 0.00419533701162551
    Epoch 9001, Loss: 0.0032952036447040957


```python
# Test the trained model
_, final_output = forward_pass(inputs.T, W1, b1, W2, b2)
print("Predicted Output after Training:\n", final_output)
```

    Predicted Output after Training:
     [[0.05464331 0.95032601 0.95006665 0.05336686]]


```python
#### Exercise: Extend to batch gradient descent rather than stochastic gradient descent.
## Which one converges faster? Plot the loss over time for both approaches.
```
