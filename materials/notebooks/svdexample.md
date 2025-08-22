---
title: "Generating a synthetic 2D dataset"
layout: note
category: "Reference Material"
permalink: /materials/notebooks/svdexample/
notebook_source: "svdexample.ipynb"
---

```python
import numpy as np
import matplotlib.pyplot as plt
```


```python


# Generating a synthetic 2D dataset
np.random.seed(0)
# Creating a dataset with a clear direction of maximum variance
x = np.random.normal(loc=0.0, scale=1.0, size=100)
y = 2.5 * x + np.random.normal(loc=0.0, scale=0.5, size=100)
data = np.vstack([x, y]).T

# Performing SVD on the zero-mean data
data_mean = np.mean(data, axis=0)
data_centered = data - data_mean
U, S, Vt = np.linalg.svd(data_centered)

# Transforming data into the principal component space
data_pca = np.dot(data_centered, Vt.T)

# Plotting the original data, mean, principal components, and transformed data
plt.figure(figsize=(12, 6))

# Original data and mean
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label='Original Data')
plt.scatter(data_mean[0], data_mean[1], color='red', label='Mean')
for i, s in enumerate(S):
    plt.arrow(data_mean[0], data_mean[1], Vt[i, 0] * s, Vt[i, 1] * s, width=0.01, color='k', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Data and Principal Components')
plt.axis('equal')  # Setting equal scaling on both axes

plt.legend()

# Transformed data
plt.subplot(1, 2, 2)
plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5, label='PCA Transformed Data')
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Transformed Data')
plt.legend()


plt.tight_layout()
plt.show()

```


    ![png](/materials/notebooks/svdexample/output_1_0.png)

```python

```
