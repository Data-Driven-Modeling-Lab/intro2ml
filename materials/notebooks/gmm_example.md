---
title: "Seed for reproducibility"
layout: note
category: "Jupyter Notebook"
permalink: /materials/notebooks/gmm_example/
notebook_source: "gmm_example.ipynb"
---

```python
import numpy as np
from scipy.stats import multivariate_normal

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic data from 3 Gaussian distributions
mean1, cov1 = [5, 5], [[1, 0], [0, 1]]
mean2, cov2 = [0, 0], [[1, 0], [0, 1]]
mean3, cov3 = [8, 1], [[1, 0], [0, 1]]

data1 = np.random.multivariate_normal(mean1, cov1, 100)
data2 = np.random.multivariate_normal(mean2, cov2, 100)
data3 = np.random.multivariate_normal(mean3, cov3, 100)

# Combine all data
data = np.vstack((data1, data2, data3))

# Number of clusters
k = 3
n, d = data.shape

```


```python

# Initialize parameters
weights = np.ones(k) / k  # Equal weights for each cluster initially
means = np.random.rand(k, d) * 10  # Random means
covariances = np.array([np.eye(d) for _ in range(k)])  # Identity covariance matrices

# EM algorithm parameters
max_iter = 100
tol = 1e-6
log_likelihoods = []

# EM algorithm
for iteration in range(max_iter):
    # E-step: compute responsibilities
    responsibilities = np.zeros((n, k))
    for i in range(k):
        responsibilities[:, i] = weights[i] * multivariate_normal.pdf(data, mean=means[i], cov=covariances[i])
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    
    # M-step: update weights, means, and covariances
    Nk = responsibilities.sum(axis=0)
    weights = Nk / n
    means = np.dot(responsibilities.T, data) / Nk[:, np.newaxis]
    covariances = np.zeros((k, d, d))
    
    for i in range(k):
        diff = data - means[i]
        covariances[i] = np.dot(responsibilities[:, i] * diff.T, diff) / Nk[i]

    # Log-likelihood calculation
    log_likelihood = np.sum(np.log(np.dot(responsibilities, weights)))
    log_likelihoods.append(log_likelihood)
    
    # Check for convergence
    if iteration > 0 and abs(log_likelihood - log_likelihoods[-2]) < tol:
        print(f'Converged at iteration {iteration}')
        break

# Output the results
print("Final weights:", weights)
print("Final means:", means)
print("Final covariances:", covariances)
```

    Converged at iteration 16
    Final weights: [0.33330769 0.33346979 0.33322252]
    Final means: [[0.1282194  0.04319516]
     [4.8851131  5.03197179]
     [7.95496913 0.87430432]]
    Final covariances: [[[ 1.07000165 -0.08054279]
      [-0.08054279  0.86404644]]
    
     [[ 0.7290481   0.02375185]
      [ 0.02375185  0.99448841]]
    
     [[ 1.04155424  0.08273123]
      [ 0.08273123  0.92681796]]]


```python

```
