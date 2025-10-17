---
title: "Problem Set 5: Unsupervised Learning"
layout: note
category: "Assignment"
permalink: /materials/assignments/hw6/
---


**Instructions:** Submit the proposal as a PDF file on the following [submission link](https://lms.aub.edu.lb/mod/assign/view.php?id=2420133)

## Problem 1: K-Means Clustering for Image Compression (30 points)

<img src="/materials/assignments/aub-pic.jpg" alt="AUB video" width="50%" />

Save the image above and reduce the number of colors it using K-means clustering. Write your own implementation of K-means clustering, and don't use any existing libraries.

a) Use 2, 8, 16, and 64 clusters and plot the original image and the compressed images. 

b) Compute/estimate the compression ratio for each case.

c) Perform the same task using SVD for image compression, using 2, 8, 16 and 64 components. 

d) Compare the SVD and K-means images. What's the difference in compression? How does the compression ratio compare to the original image?


## Problem 2: PCA and nonlinear dimensionality reduction (70 points)

Find and load the MNIST dataset.  

a) Apply PCA to reduce the dimensionality of the data. Use 2, 8, 16, and 64 components. Plot the original image and the reconstructed images. Compute the compression ratio for each case.

b) Is compressing the data from the original 784 dimensions to 2 dimensions equivalent to doing it in two stages: first reducing the dimensionality to 100 and then to 2? 1) Plot the original image and the reconstructed images for both cases. 2) Compute the compression ratio for each case. 3) Prove that the two approaches are equivalent.

c) Given the normalized data matrix $X$, and the covariance matrix $\Sigma = \frac{1}{n-1} X^T X$, show that the eigenvalue decomposition $\Sigma = V \Lambda V^T$ on the covariance matrix is equivalent to the singular value decomposition of the data matrix $X = U \Sigma V^T$.

d) Perform a nonlinear dimensionality reduction using an autoencoder architecture (on the flattened images). 1) Plot the original image and the reconstructed images for 5 random samples. 2) Does the data in the reduced dimension space form meaningful clusters? Answer the question through a classification task on the reduced dimension space, and explain why this is a good or bad idea.

e) Repeat the same task using convolutional layers in the encoder and decoder. Do you observe better clustering in the latent space?

f) Repeat the same task using t-SNE and hierarchical clustering (using scikit-learn). Do you observe better clustering in the latent space? 

g) Plot the data in reduced spaces from all methods next to each other for comparison, and discuss the differences. What do you think works best for this dataset?