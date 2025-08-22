---
title: "Assume X is your data matrix, with each column being a feature and each row a data point"
layout: note
category: "Reference Material"
permalink: /materials/notebooks/pca_demo/
notebook_source: "pca_demo.ipynb"
---

```python
import numpy as np

# Assume X is your data matrix, with each column being a feature and each row a data point
# Stack your data vectors x_1, x_2, ..., x_n as rows in X (shape: n_samples x n_features)

# Step 1: Center the data by subtracting the mean of each feature
X_centered = X - np.mean(X, axis=0)

# Compute the covariance matrix
Sigma = X_centered.T @ X_centered

# Find eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(Sigma)

# Step 4: Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# The eigenvalues and eigenvectors are now sorted in descending order of variance explained
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

```
