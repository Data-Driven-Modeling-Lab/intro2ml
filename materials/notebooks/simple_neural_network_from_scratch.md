---
title: "Set random seed for reproducibility"
layout: note
category: "Jupyter Notebook"
permalink: /materials/notebooks/simple_neural_network_from_scratch/
notebook_source: "simple_neural_network_from_scratch.ipynb"
---

```python
import numpy as np
```

### Step 2: Define the Activation Function


```python

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sig):
    return sig * (1 - sig)  # Derivative of the sigmoid function

```

### Step 3: Initialize Weights and Biases


```python

np.random.seed(42)

# Initialize weights
input_layer_neurons = 2  # Number of input neurons (features)
hidden_layer_neurons = 2  # Number of hidden neurons
output_neurons = 1  # Number of output neurons

# Weights and biases
weights_input_hidden = np.random.rand(input_layer_neurons, hidden_layer_neurons)
weights_hidden_output = np.random.rand(hidden_layer_neurons, output_neurons)

bias_hidden = np.random.rand(1, hidden_layer_neurons)
bias_output = np.random.rand(1, output_neurons)

print("Initial Weights (Input to Hidden):\n", weights_input_hidden)
print("\n Initial Weights (Hidden to Output):\n", weights_hidden_output)

```

    Initial Weights (Input to Hidden):
     [[0.37454012 0.95071431]
     [0.73199394 0.59865848]]
    
     Initial Weights (Hidden to Output):
     [[0.15601864]
     [0.15599452]]


### Step 4: Forward Pass


```python

def forward_pass(inputs):
    # Hidden layer calculations
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    # Output layer calculations
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)

    return hidden_layer_output, output_layer_output

```

### Step 5: Backpropagation


```python

def backpropagation(inputs, hidden_output, predicted_output, actual_output, learning_rate):
    global weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

    # Calculate output layer error (actual - predicted)
    error_output = actual_output - predicted_output

    # Calculate gradient for weights between hidden and output layer
    d_output = error_output * sigmoid_derivative(predicted_output)

    # Calculate error for hidden layer (by backpropagating the error)
    error_hidden = d_output.dot(weights_hidden_output.T)
    
    # Calculate gradient for weights between input and hidden layer
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and biases
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += inputs.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

```

### Step 6: Training Loop


```python

# Sample data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])  # XOR problem

# Training parameters
epochs = 10000
learning_rate = 0.1

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_output, predicted_output = forward_pass(inputs)

    # Backpropagation
    backpropagation(inputs, hidden_output, predicted_output, outputs, learning_rate)

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean((outputs - predicted_output) ** 2)
        print(f'Epoch {epoch+1}, Loss: {loss}')

```

    Epoch 1, Loss: 0.287974821321425
    Epoch 1001, Loss: 0.24943329766543199
    Epoch 2001, Loss: 0.24567537147115226
    Epoch 3001, Loss: 0.21996241841579695
    Epoch 4001, Loss: 0.1621992454420141
    Epoch 5001, Loss: 0.05270887579146114
    Epoch 6001, Loss: 0.016926012420416685
    Epoch 7001, Loss: 0.00891778531419988
    Epoch 8001, Loss: 0.00584454666369325
    Epoch 9001, Loss: 0.0042817900236595905


### Step 7: Test the Neural Network


```python

# Test the trained model
_, final_output = forward_pass(inputs)
print("Predicted Output after Training:\n", final_output)

```

    Predicted Output after Training:
     [[0.06028403]
     [0.9444784 ]
     [0.9443732 ]
     [0.05996465]]

