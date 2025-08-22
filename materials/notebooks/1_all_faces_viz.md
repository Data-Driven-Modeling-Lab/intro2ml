---
title: "1 All Faces Viz"
layout: note
category: "Reference Material"
permalink: /materials/notebooks/1_all_faces_viz/
notebook_source: "1_all_faces_viz.ipynb"
---

```python
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 18})

mat_contents = scipy.io.loadmat('allFaces.mat')
faces = mat_contents['faces']
m = int(mat_contents['m'])
n = int(mat_contents['n'])
nfaces = np.ndarray.flatten(mat_contents['nfaces'])

allPersons = np.zeros((n*6,m*6))
count = 0

for j in range(6):
    for k in range(6):
        allPersons[j*n : (j+1)*n, k*m : (k+1)*m] = np.reshape(faces[:,np.sum(nfaces[:count])],(m,n)).T
        count += 1
        
img = plt.imshow(allPersons)
img.set_cmap('gray')
plt.axis('off')
plt.show()
```

    /var/folders/q4/_twpfpf54f3f6s17s74p67tc0000gp/T/ipykernel_30162/1260173006.py:10: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
      m = int(mat_contents['m'])
    /var/folders/q4/_twpfpf54f3f6s17s74p67tc0000gp/T/ipykernel_30162/1260173006.py:11: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
      n = int(mat_contents['n'])


    
![png](/materials/notebooks/1_all_faces_viz/output_0_1.png)
    


```python
print(m , n)
```

    168 192


```python
for person in range(len(nfaces)):
    subset = faces[:,sum(nfaces[:person]) : sum(nfaces[:(person+1)])]
    allFaces = np.zeros((n*8,m*8))
    
    count = 0
    
    for j in range(8):
        for k in range(8):
            if count < nfaces[person]:
                allFaces[j*n:(j+1)*n,k*m:(k+1)*m] = np.reshape(subset[:,count],(m,n)).T
                count += 1
                
    img = plt.imshow(allFaces)
    img.set_cmap('gray')
    plt.axis('off')
    plt.show()
```


    
![png](/materials/notebooks/1_all_faces_viz/output_2_0.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_1.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_2.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_3.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_4.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_5.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_6.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_7.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_8.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_9.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_10.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_11.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_12.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_13.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_14.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_15.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_16.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_17.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_18.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_19.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_20.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_21.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_22.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_23.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_24.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_25.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_26.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_27.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_28.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_29.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_30.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_31.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_32.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_33.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_34.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_35.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_36.png)
    


    
![png](/materials/notebooks/1_all_faces_viz/output_2_37.png)
    


```python

```
