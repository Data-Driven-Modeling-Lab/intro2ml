---
title: "1. Download the MNIST dataset"
layout: note
category: "Reference Material"
permalink: /materials/notebooks/mnist_example/
notebook_source: "mnist_example.ipynb"
---

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

# 1. Download the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data  # Feature matrix (70,000 samples of 784 features)
y = mnist.target.astype(int)  # Labels (digits 0-9)

```


    
![png](/materials/notebooks/mnist_example/output_0_0.png)
    


```python

# 2. Perform PCA to reduce the dimensionality to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. Plot the first two principal components
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7, s=15)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of MNIST Dataset')
plt.colorbar(scatter, ticks=range(10), label='Digit Label')
plt.grid(True)
plt.show()
```


```python
from sklearn.mixture import GaussianMixture

# 3. Fit a Gaussian Mixture Model with 10 components
gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)
gmm.fit(X_pca)

# 4. Retrieve the means and covariances
means = gmm.means_  # Shape (10, 2)
covariances = gmm.covariances_  # Shape (10, 2, 2)

# 5. Compute standard deviations from covariances
stds = np.sqrt(np.array([np.diag(cov) for cov in covariances]))  # Shape (10, 2)

# 6. Print the means and standard deviations
print("Means of the clusters:")
for idx, mean in enumerate(means):
    print(f"Cluster {idx}: Mean = {mean}")

print("\nStandard deviations of the clusters:")
for idx, std in enumerate(stds):
    print(f"Cluster {idx}: Std Dev = {std}")

# 7. Optional: Plot the GMM clusters with ellipses
def plot_gmm(gmm, X, label=True, ax=None):
    import matplotlib as mpl
    ax = ax or plt.gca()
    labels = gmm.predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=15, cmap='viridis', alpha=0.5)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=15, alpha=0.5)
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor, ax=ax)

def draw_ellipse(position, covariance, ax=None, **kwargs):
    from matplotlib.patches import Ellipse
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        width, height = 2 * np.sqrt(covariance)
        angle = 0
    for nsig in range(1, 4):  # 1 to 3 standard deviations
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

plt.figure(figsize=(12, 10))
plot_gmm(gmm, X_pca)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('GMM Clusters in PCA-transformed MNIST Data')
plt.grid(True)
plt.show()
```

    Means of the clusters:
    Cluster 0: Mean = [ 300.54794659 -660.92240369]
    Cluster 1: Mean = [-387.65974065  481.74359792]
    Cluster 2: Mean = [81.24461923  3.09931913]
    Cluster 3: Mean = [-893.27200304 -461.45418997]
    Cluster 4: Mean = [ 593.98676639 -153.96015551]
    Cluster 5: Mean = [1239.62754088 -218.64778218]
    Cluster 6: Mean = [424.82633416 570.08093663]
    Cluster 7: Mean = [-506.72302081  -92.98536502]
    Cluster 8: Mean = [-181.23886685 -403.95786527]
    Cluster 9: Mean = [-26.87774701 779.78471607]
    
    Standard deviations of the clusters:
    Cluster 0: Std Dev = [313.85594823 259.21437006]
    Cluster 1: Std Dev = [180.0142689  191.50979434]
    Cluster 2: Std Dev = [211.18878722 250.5613713 ]
    Cluster 3: Std Dev = [ 78.54624225 191.95067188]
    Cluster 4: Std Dev = [254.30619766 226.18503271]
    Cluster 5: Std Dev = [351.423388   236.18368818]
    Cluster 6: Std Dev = [282.28554376 303.0627887 ]
    Cluster 7: Std Dev = [199.13299941 210.05289074]
    Cluster 8: Std Dev = [246.49344149 227.05246774]
    Cluster 9: Std Dev = [217.81045166 181.02033043]


    /var/folders/q4/_twpfpf54f3f6s17s74p67tc0000gp/T/ipykernel_93309/1072052865.py:47: MatplotlibDeprecationWarning: Passing the angle parameter of __init__() positionally is deprecated since Matplotlib 3.6; the parameter will become keyword-only two minor releases later.
      ax.add_patch(Ellipse(position, nsig * width, nsig * height,


    
![png](/materials/notebooks/mnist_example/output_2_2.png)
    

