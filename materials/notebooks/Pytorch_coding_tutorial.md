---
title: "Create a tensor"
layout: note
category: "Jupyter Notebook"
permalink: /materials/notebooks/Pytorch_coding_tutorial/
notebook_source: "Pytorch_coding_tutorial.ipynb"
---

If you don’t have PyTorch installed yet, you can install it by running:


```python
pip install torch torchvision
```

    Requirement already satisfied: torch in /opt/anaconda3/lib/python3.9/site-packages (2.2.2)
    Collecting torchvision
      Downloading torchvision-0.17.2-cp39-cp39-macosx_10_13_x86_64.whl.metadata (6.6 kB)
    Requirement already satisfied: filelock in /opt/anaconda3/lib/python3.9/site-packages (from torch) (3.6.0)
    Requirement already satisfied: typing-extensions>=4.8.0 in /opt/anaconda3/lib/python3.9/site-packages (from torch) (4.12.2)
    Requirement already satisfied: sympy in /opt/anaconda3/lib/python3.9/site-packages (from torch) (1.11.1)
    Requirement already satisfied: networkx in /opt/anaconda3/lib/python3.9/site-packages (from torch) (2.8.4)
    Requirement already satisfied: jinja2 in /opt/anaconda3/lib/python3.9/site-packages (from torch) (3.1.2)
    Requirement already satisfied: fsspec in /opt/anaconda3/lib/python3.9/site-packages (from torch) (2024.3.1)
    Requirement already satisfied: numpy in /opt/anaconda3/lib/python3.9/site-packages (from torchvision) (1.26.4)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /opt/anaconda3/lib/python3.9/site-packages (from torchvision) (9.2.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/anaconda3/lib/python3.9/site-packages (from jinja2->torch) (2.1.3)
    Requirement already satisfied: mpmath>=0.19 in /opt/anaconda3/lib/python3.9/site-packages (from sympy->torch) (1.2.1)
    Downloading torchvision-0.17.2-cp39-cp39-macosx_10_13_x86_64.whl (1.7 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m923.2 kB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hInstalling collected packages: torchvision
    Successfully installed torchvision-0.17.2
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.1.1[0m[39;49m -> [0m[32;49m24.2[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.


### PyTorch is like NumPy, but with strong GPU acceleration


```python
import torch

tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Check the shape
print(tensor.shape)

# Perform operations
tensor_add = tensor + 2
tensor_mul = tensor * 3

# Display results
print("Original Tensor:\n", tensor)
print("Tensor after addition:\n", tensor_add)
print("Tensor after multiplication:\n", tensor_mul)
```

    torch.Size([2, 2])
    Original Tensor:
     tensor([[1., 2.],
            [3., 4.]])
    Tensor after addition:
     tensor([[3., 4.],
            [5., 6.]])
    Tensor after multiplication:
     tensor([[ 3.,  6.],
            [ 9., 12.]])


```python

```


```python
### Exercise 1: Try creating tensors of different shapes and performing operations like addition, multiplication, etc.
```

### Gradients in PyTorch

PyTorch automatically calculates gradients during backpropagation using autograd.


```python
# Create a tensor with gradient tracking enabled
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Define a simple function
y = x[0]**2 + x[1]**2

# Perform backpropagation
y.backward()

# Access the gradient
print("Gradient of x[0]:", x.grad[0])
print("Gradient of x[1]:", x.grad[1])
```

    Gradient of x[0]: tensor(4.)
    Gradient of x[1]: tensor(6.)


```python
### Exercise 2: Play around with different functions to see how gradients are calculated.
```

### Building a Simple Neural Network

Now, let’s define a basic neural network using PyTorch’s nn.Module


```python
import torch.nn as nn

# Define a simple fully connected neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 3)  # Input layer
        self.fc2 = nn.Linear(3, 1)  # Output layer

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the network and print the structure
model = SimpleNN()
print(model)
```

    SimpleNN(
      (fc1): Linear(in_features=2, out_features=3, bias=True)
      (fc2): Linear(in_features=3, out_features=1, bias=True)
    )


```python
### Exercise 3: Try changing the input and output dimensions to understand how they affect the network structure.
```

### Forward Pass and Loss Function

We’ll pass some data through the network and calculate a loss.


```python
# Sample data
input = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
true_output = torch.tensor([[5.0], [7.0]])

# Forward pass
predicted_output = model(input)

# Define mean squared error loss
criterion = nn.MSELoss()

# Calculate loss
loss = criterion(predicted_output, true_output)
print("Loss:", loss.item())
```

    Loss: 33.52677536010742


```python
### Exercise 4: Try changing the input and output values to see how the loss changes.
```


```python
model.parameters()
```


    <generator object Module.parameters at 0x7f9c20477c10>


### Optimizers and Training Loop

We’ll now define an optimizer and train the network using a basic training loop.


```python
import torch.optim as optim

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):  # 100 epochs
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    predicted_output = model(input)
    
    # Calculate loss
    loss = criterion(predicted_output, true_output)
    
    # Backpropagation
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
```

    Epoch 1, Loss: 33.52677536010742
    Epoch 11, Loss: 11.799346923828125
    Epoch 21, Loss: 3.8020379543304443
    Epoch 31, Loss: 1.4266102313995361
    Epoch 41, Loss: 0.8570533394813538
    Epoch 51, Loss: 0.7266203761100769
    Epoch 61, Loss: 0.6920912861824036
    Epoch 71, Loss: 0.6777915358543396
    Epoch 81, Loss: 0.6676491498947144
    Epoch 91, Loss: 0.658282458782196


```python
### Exercise 5: Adjust the learning rate, the number of epochs, or try different optimizers (like Adam) to see how it affects the training process. 
### Look up the various optimizers available in PyTorch and and research how they differ. 
```

### Evaluate the Trained Model

Finally, we’ll evaluate the model on the test dataset to see how well it performs.


```python
# Test the model on new data

test_predicted = model(test_data)
print("Predicted value:", test_predicted.item())
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/q4/_twpfpf54f3f6s17s74p67tc0000gp/T/ipykernel_37383/193933764.py in <module>
          1 # Test the model on new data
          2 
    ----> 3 test_predicted = model(test_data)
          4 print("Predicted value:", test_predicted.item())


    NameError: name 'test_data' is not defined

