---
title: "X = X.T"
layout: note
category: "Reference Material"
permalink: /materials/notebooks/0_image_approx/
notebook_source: "0_image_approx.ipynb"
---

```python
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [4, 7]


A = imread('dog.jpg')
X = np.mean(A, -1); # Convert RGB to grayscale

img = plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')
plt.show()
```


    ![png](/materials/notebooks/0_image_approx/output_1_0.png)

```python
# X = X.T
U_tild, S_t, VT_tild = np.linalg.svd(X, full_matrices=False)
S_tild = np.diag(S_t)
```


```python
print(U_tild.shape)
print(S_tild.shape)
print(VT_tild.shape)
```

    (2000, 1500)
    (1500, 1500)
    (1500, 1500)


```python
U, S, VT = np.linalg.svd(X,full_matrices=True)
S = np.diag(S)

print(U.shape)
print(S.shape)
print(VT.shape)
```

    (2000, 2000)
    (1500, 1500)
    (1500, 1500)


```python
%matplotlib inline

j = 0
for r in (5, 20, 100):
    # Construct approximate image
    Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    plt.figure(j+1, figsize=(4, 7))
    j += 1
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r = ' + str(r))
    plt.show()
```

    /var/folders/q4/_twpfpf54f3f6s17s74p67tc0000gp/T/ipykernel_30142/2514985480.py:9: UserWarning: Attempt to set non-positive ylim on a log-scaled axis will be ignored.
      img = plt.imshow(Xapprox)


    
![png](/materials/notebooks/0_image_approx/output_4_1.png)
    


    
![png](/materials/notebooks/0_image_approx/output_4_2.png)
    


    
![png](/materials/notebooks/0_image_approx/output_4_3.png)
    


    
![png](/materials/notebooks/0_image_approx/output_4_4.png)
    


    
![png](/materials/notebooks/0_image_approx/output_4_5.png)
    


```python
r = 20
print('x', X.shape)
print('u', U[:, :r].shape)
print('s', S[:r, :r].shape)
print('vt', VT[:r, :].shape)
```

    x (2000, 1500)
    u (2000, 20)
    s (20, 20)
    vt (20, 1500)


```python
## f_ch01_ex02_2
%matplotlib widget

plt.figure(1, figsize=(5, 3))
plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.show()

plt.figure(2, figsize=(5, 3))
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('Singular Values: Cumulative Sum')
plt.show()
```


    ![png](/materials/notebooks/0_image_approx/output_2_1.png)

    ![png](/materials/notebooks/0_image_approx/output_3_2.png)

