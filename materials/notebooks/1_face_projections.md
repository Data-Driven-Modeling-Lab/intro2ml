---
title: "We use the first 36 people for training data"
layout: note
category: "Reference Material"
permalink: /materials/notebooks/1_face_projections/
notebook_source: "1_face_projections.ipynb"
---

```python
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

mat_contents = scipy.io.loadmat('allFaces.mat')
faces = mat_contents['faces']
m = int(mat_contents['m'])
n = int(mat_contents['n'])
nfaces = np.ndarray.flatten(mat_contents['nfaces'])

# We use the first 36 people for training data
trainingFaces = faces[:,:np.sum(nfaces[:36])]
avgFace = np.mean(trainingFaces, axis=1) # size n*m by 1

# Compute eigenfaces on mean-subtracted training data
X = trainingFaces - np.tile(avgFace,(trainingFaces.shape[1],1)).T
U, S, VT = np.linalg.svd(X, full_matrices=False)

fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
img_avg = ax1.imshow(np.reshape(avgFace,(m,n)).T)
img_avg.set_cmap('gray')
plt.axis('off')

ax2 = fig1.add_subplot(122)
img_u1 = ax2.imshow(np.reshape(U[:,0],(m,n)).T)
img_u1.set_cmap('gray')
plt.axis('off')

plt.show()
```

    /var/folders/wq/rd7c2mhn7fs9y313qjs3c58r0000gn/T/ipykernel_11190/592215803.py:10: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
      m = int(mat_contents['m'])
    /var/folders/wq/rd7c2mhn7fs9y313qjs3c58r0000gn/T/ipykernel_11190/592215803.py:11: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
      n = int(mat_contents['n'])


    
![png](/materials/notebooks/1_face_projections/output_0_1.png)
    


```python
nfaces
```


    array([64, 62, 64, 64, 62, 64, 64, 64, 64, 64, 60, 59, 60, 63, 62, 63, 63,
           64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
           64, 64, 64, 64], dtype=uint8)


```python
## Now show eigenface reconstruction of image that was omitted from test set

testFace = faces[:,np.sum(nfaces[:36])] # First face of person 37
plt.imshow(np.reshape(testFace,(m,n)).T)
plt.set_cmap('gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

testFaceMS = testFace - avgFace
r_list = [25, 50, 100, 200, 400, 800, 1600]

for r in r_list:
    reconFace = avgFace + U[:,:r]  @ U[:,:r].T @ testFaceMS
    img = plt.imshow(np.reshape(reconFace,(m,n)).T)
    img.set_cmap('gray')
    plt.title('r = ' + str(r))
    plt.axis('off')
    plt.show()
```


    
![png](/materials/notebooks/1_face_projections/output_2_0.png)
    


    
![png](/materials/notebooks/1_face_projections/output_2_1.png)
    


    
![png](/materials/notebooks/1_face_projections/output_2_2.png)
    


    
![png](/materials/notebooks/1_face_projections/output_2_3.png)
    


    
![png](/materials/notebooks/1_face_projections/output_2_4.png)
    


    
![png](/materials/notebooks/1_face_projections/output_2_5.png)
    


    
![png](/materials/notebooks/1_face_projections/output_2_6.png)
    


    
![png](/materials/notebooks/1_face_projections/output_2_7.png)
    


```python
## Project person 2 and 7 onto PC5 and PC6

P1num = 1 # Person number 2
P2num = 2 # Person number 7

P1 = faces[:,np.sum(nfaces[:(P1num-1)]):np.sum(nfaces[:P1num])]
P2 = faces[:,np.sum(nfaces[:(P2num-1)]):np.sum(nfaces[:P2num])]

P1 = P1 - np.tile(avgFace,(P1.shape[1],1)).T
P2 = P2 - np.tile(avgFace,(P2.shape[1],1)).T

PCAmodes = [1, 0] # Project onto PCA modes 5 and 6
PCACoordsP1 = U[:,PCAmodes-np.ones_like(PCAmodes)].T @ P1
PCACoordsP2 = U[:,PCAmodes-np.ones_like(PCAmodes)].T @ P2

plt.plot(PCACoordsP1[0,:],PCACoordsP1[1,:],'d',color='k',label='Person '+str(P1num))
plt.plot(PCACoordsP2[0,:],PCACoordsP2[1,:],'^',color='r',label='Person '+str(P2num))

plt.legend()
plt.show()
```


    
![png](/materials/notebooks/1_face_projections/output_3_0.png)
    


```python

```
