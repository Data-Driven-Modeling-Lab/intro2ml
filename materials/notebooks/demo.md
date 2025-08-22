---
title: "Define the neural network"
layout: note
category: "Jupyter Notebook"
permalink: /materials/notebooks/demo/
notebook_source: "demo.ipynb"
---

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Input layer (28x28 pixels -> 128 hidden units)
        self.fc1 = nn.Linear(28 * 28, 128)
        # Hidden layer (128 -> 64 hidden units)
        self.fc2 = nn.Linear(128, 64)
        # Output layer (64 -> 10 classes for digits)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Flatten the input (batch_size, 28*28)
        x = x.view(-1, 28 * 28)
        # Apply ReLU activation on each hidden layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer (no activation, since we'll use softmax later)
        x = self.fc3(x)
        return x

# Load the MNIST dataset (handwritten digits)
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Initialize the neural network, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # Cross entropy for classification
optimizer = optim.SGD(model.parameters(), lr=0.01)  # SGD optimizer

# Training loop
for epoch in range(5):  # 5 epochs
    running_loss = 0.0
    for images, labels in train_loader:
        # Zero the gradients from the previous iteration
        optimizer.zero_grad()
        
        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        
        # Calculate the loss
        loss = criterion(outputs, labels)
        
        # Backward pass: compute gradient of the loss w.r.t. model parameters
        loss.backward()
        
        # Perform a single optimization step (update weights)
        optimizer.step()
        
        # Print loss
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/5], Loss: {running_loss/len(train_loader):.4f}")
```

    Epoch [1/5], Loss: 1.7882
    Epoch [2/5], Loss: 0.5892
    Epoch [3/5], Loss: 0.4007
    Epoch [4/5], Loss: 0.3472
    Epoch [5/5], Loss: 0.3170

