---
title: "Matrices and Matrix Operations"
layout: note
category: "Jupyter Notebook"
permalink: /materials/notebooks/02_matrices_and_operations/
notebook_source: "02_matrices_and_operations.ipynb"
---


**Based on CS229 Linear Algebra Review - Section 2**

Matrices are fundamental to machine learning. They represent datasets, transformations, and relationships between variables. In this notebook, we'll explore matrix operations and their Python implementations.


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn-v0_8')
np.random.seed(42)
```

## 1. What is a Matrix?

### Mathematical Definition
A matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ is a rectangular array of real numbers with $m$ rows and $n$ columns:

$$\mathbf{A} = \begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}$$

### In Machine Learning
- **Data matrix**: Each row is a sample, each column is a feature
- **Transformation**: Linear mappings between vector spaces
- **Parameters**: Model weights and coefficients


```python
# Creating matrices in NumPy

# Method 1: From a list of lists
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"Matrix A:")
print(A)
print(f"Shape: {A.shape} (2 rows, 3 columns)")
print(f"Size: {A.size} elements")

# Method 2: Using NumPy functions
B = np.zeros((3, 3))  # 3x3 matrix of zeros
C = np.ones((2, 4))   # 2x4 matrix of ones
D = np.eye(3)         # 3x3 identity matrix
E = np.random.rand(2, 3)  # 2x3 random matrix

print(f"\nZeros matrix B:")
print(B)
print(f"\nOnes matrix C:")
print(C)
print(f"\nIdentity matrix D:")
print(D)
print(f"\nRandom matrix E:")
print(E)
```

    Matrix A:
    [[1 2 3]
     [4 5 6]]
    Shape: (2, 3) (2 rows, 3 columns)
    Size: 6 elements
    
    Zeros matrix B:
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    
    Ones matrix C:
    [[1. 1. 1. 1.]
     [1. 1. 1. 1.]]
    
    Identity matrix D:
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    
    Random matrix E:
    [[0.37454012 0.95071431 0.73199394]
     [0.59865848 0.15601864 0.15599452]]


## 2. Matrix Indexing and Slicing

### Accessing Elements
- $a_{ij}$ is the element in row $i$, column $j$ (1-indexed in math, 0-indexed in Python)
- Rows and columns can be extracted as vectors


```python
# Matrix indexing and slicing
M = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

print(f"Matrix M:")
print(M)
print()

# Access single element
print(f"Element at position (0,2): {M[0, 2]}")
print(f"Element at position (2,1): {M[2, 1]}")

# Access entire rows
print(f"\nFirst row: {M[0, :]}")
print(f"Second row: {M[1, :]}")

# Access entire columns
print(f"\nFirst column: {M[:, 0]}")
print(f"Third column: {M[:, 2]}")

# Submatrices
print(f"\nTop-left 2x2 submatrix:")
print(M[0:2, 0:2])

print(f"\nLast two rows, last two columns:")
print(M[1:, 2:])
```

    Matrix M:
    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]]
    
    Element at position (0,2): 3
    Element at position (2,1): 10
    
    First row: [1 2 3 4]
    Second row: [5 6 7 8]
    
    First column: [1 5 9]
    Third column: [ 3  7 11]
    
    Top-left 2x2 submatrix:
    [[1 2]
     [5 6]]
    
    Last two rows, last two columns:
    [[ 7  8]
     [11 12]]


## 3. Matrix Addition and Scalar Multiplication

### Mathematical Definition
For matrices $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$:

**Addition**: $(\mathbf{A} + \mathbf{B})_{ij} = a_{ij} + b_{ij}$

**Scalar multiplication**: $(\alpha \mathbf{A})_{ij} = \alpha a_{ij}$

**Note**: Matrices must have the same dimensions for addition.


```python
# Matrix addition and scalar multiplication
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(f"Matrix A:")
print(A)
print(f"\nMatrix B:")
print(B)

# Matrix addition
C = A + B
print(f"\nA + B:")
print(C)

# Matrix subtraction
D = A - B
print(f"\nA - B:")
print(D)

# Scalar multiplication
E = 3 * A
print(f"\n3 * A:")
print(E)

# Element-wise operations
F = A * B  # Element-wise multiplication (Hadamard product)
print(f"\nA ⊙ B (element-wise multiplication):")
print(F)

# Visualize matrix operations
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
matrices = [A, B, C, D, E, F]
titles = ['A', 'B', 'A + B', 'A - B', '3 × A', 'A ⊙ B']

for i, (mat, title) in enumerate(zip(matrices, titles)):
    row, col = i // 3, i % 3
    im = axes[row, col].imshow(mat, cmap='viridis', aspect='equal')
    axes[row, col].set_title(title)
    
    # Add text annotations
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            axes[row, col].text(c, r, f'{mat[r,c]:.0f}', 
                               ha='center', va='center', 
                               color='white' if mat[r,c] < mat.max()/2 else 'black')
    
    axes[row, col].set_xticks([])
    axes[row, col].set_yticks([])

plt.tight_layout()
plt.show()
```

    Matrix A:
    [[1 2]
     [3 4]]
    
    Matrix B:
    [[5 6]
     [7 8]]
    
    A + B:
    [[ 6  8]
     [10 12]]
    
    A - B:
    [[-4 -4]
     [-4 -4]]
    
    3 * A:
    [[ 3  6]
     [ 9 12]]
    
    A ⊙ B (element-wise multiplication):
    [[ 5 12]
     [21 32]]


    /var/folders/q4/_twpfpf54f3f6s17s74p67tc0000gp/T/ipykernel_87550/1003322807.py:52: UserWarning: Glyph 8857 (\N{CIRCLED DOT OPERATOR}) missing from current font.
      plt.tight_layout()
    /opt/anaconda3/lib/python3.9/site-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 8857 (\N{CIRCLED DOT OPERATOR}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)


    
![png](/materials/notebooks/02_matrices_and_operations/output_7_2.png)
    


## 4. Matrix Multiplication

### Mathematical Definition
For $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n \times p}$:

$$(\mathbf{AB})_{ij} = \sum_{k=1}^n a_{ik} b_{kj}$$

**Key requirements**:
- Number of columns in $\mathbf{A}$ must equal number of rows in $\mathbf{B}$
- Result is $m \times p$ matrix

### Geometric Interpretation
Matrix multiplication represents composition of linear transformations.


```python
# Matrix multiplication
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

print(f"Matrix A (2×3):")
print(A)
print(f"\nMatrix B (3×2):")
print(B)

# Matrix multiplication using @
C = A @ B
print(f"\nA @ B (2×2):")
print(C)

# Alternative: using np.dot
C_alt = np.dot(A, B)
print(f"\nUsing np.dot (same result):")
print(C_alt)

# Manual calculation for verification
print(f"\nManual calculation:")
print(f"C[0,0] = A[0,:] · B[:,0] = {A[0,:]} · {B[:,0]} = {np.dot(A[0,:], B[:,0])}")
print(f"C[0,1] = A[0,:] · B[:,1] = {A[0,:]} · {B[:,1]} = {np.dot(A[0,:], B[:,1])}")
print(f"C[1,0] = A[1,:] · B[:,0] = {A[1,:]} · {B[:,0]} = {np.dot(A[1,:], B[:,0])}")
print(f"C[1,1] = A[1,:] · B[:,1] = {A[1,:]} · {B[:,1]} = {np.dot(A[1,:], B[:,1])}")

# Dimension compatibility check
print(f"\nDimension compatibility:")
print(f"A.shape = {A.shape}, B.shape = {B.shape}")
print(f"A has {A.shape[1]} columns, B has {B.shape[0]} rows")
print(f"Since {A.shape[1]} == {B.shape[0]}, multiplication is valid")
print(f"Result shape: ({A.shape[0]}, {B.shape[1]}) = {C.shape}")
```

    Matrix A (2×3):
    [[1 2 3]
     [4 5 6]]
    
    Matrix B (3×2):
    [[ 7  8]
     [ 9 10]
     [11 12]]
    
    A @ B (2×2):
    [[ 58  64]
     [139 154]]
    
    Using np.dot (same result):
    [[ 58  64]
     [139 154]]
    
    Manual calculation:
    C[0,0] = A[0,:] · B[:,0] = [1 2 3] · [ 7  9 11] = 58
    C[0,1] = A[0,:] · B[:,1] = [1 2 3] · [ 8 10 12] = 64
    C[1,0] = A[1,:] · B[:,0] = [4 5 6] · [ 7  9 11] = 139
    C[1,1] = A[1,:] · B[:,1] = [4 5 6] · [ 8 10 12] = 154
    
    Dimension compatibility:
    A.shape = (2, 3), B.shape = (3, 2)
    A has 3 columns, B has 3 rows
    Since 3 == 3, multiplication is valid
    Result shape: (2, 2) = (2, 2)


## 5. Matrix-Vector Multiplication

### Mathematical Definition
For $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{x} \in \mathbb{R}^n$:

$$\mathbf{Ax} = \begin{bmatrix} 
\mathbf{a}_1^T \mathbf{x} \\
\mathbf{a}_2^T \mathbf{x} \\
\vdots \\
\mathbf{a}_m^T \mathbf{x}
\end{bmatrix}$$

where $\mathbf{a}_i^T$ is the $i$-th row of $\mathbf{A}$.

### In Machine Learning
This is fundamental for:
- Linear regression: $\hat{y} = \mathbf{Ax}$
- Neural networks: forward propagation
- Feature transformations


```python
# Matrix-vector multiplication
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
x = np.array([1, 2, 1])

print(f"Matrix A:")
print(A)
print(f"\nVector x: {x}")

# Matrix-vector multiplication
y = A @ x
print(f"\nA @ x = {y}")

# Show row-wise computation
print(f"\nRow-wise computation:")
for i in range(A.shape[0]):
    row_product = np.dot(A[i, :], x)
    print(f"Row {i}: {A[i, :]} · {x} = {row_product}")

# Visualize as linear transformation
# Let's use a 2D example for better visualization
A_2d = np.array([[2, 1],
                 [1, 2]])
x_2d = np.array([1, 1])
y_2d = A_2d @ x_2d

plt.figure(figsize=(10, 5))

# Plot original and transformed vectors
plt.subplot(1, 2, 1)
plt.arrow(0, 0, x_2d[0], x_2d[1], head_width=0.1, head_length=0.1, 
          fc='blue', ec='blue', label='Original x')
plt.arrow(0, 0, y_2d[0], y_2d[1], head_width=0.1, head_length=0.1, 
          fc='red', ec='red', label='Transformed Ax')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.legend()
plt.title('Linear Transformation')
plt.xlabel('x')
plt.ylabel('y')

# Show the computation
plt.subplot(1, 2, 2)
computation_text = f"""Matrix A:
{A_2d}

Vector x: {x_2d}

A @ x = {y_2d}

Computation:
y[0] = {A_2d[0,0]}×{x_2d[0]} + {A_2d[0,1]}×{x_2d[1]} = {y_2d[0]}
y[1] = {A_2d[1,0]}×{x_2d[0]} + {A_2d[1,1]}×{x_2d[1]} = {y_2d[1]}"""
plt.text(0.1, 0.5, computation_text, transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='center', fontfamily='monospace')
plt.axis('off')
plt.title('Computation Details')

plt.tight_layout()
plt.show()
```

    Matrix A:
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    
    Vector x: [1 2 1]
    
    A @ x = [ 8 20 32]
    
    Row-wise computation:
    Row 0: [1 2 3] · [1 2 1] = 8
    Row 1: [4 5 6] · [1 2 1] = 20
    Row 2: [7 8 9] · [1 2 1] = 32


    
![png](/materials/notebooks/02_matrices_and_operations/output_11_1.png)
    


## 6. Special Matrices

### Identity Matrix
The identity matrix $\mathbf{I} \in \mathbb{R}^{n \times n}$ has 1s on the diagonal and 0s elsewhere:
$$\mathbf{I} = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}$$

**Property**: $\mathbf{AI} = \mathbf{IA} = \mathbf{A}$ for any compatible matrix $\mathbf{A}$.

### Diagonal Matrix
A matrix with non-zero elements only on the diagonal.

### Symmetric Matrix
A matrix where $\mathbf{A} = \mathbf{A}^T$ (i.e., $a_{ij} = a_{ji}$).


```python
# Special matrices
print("Special Matrices:")
print("=" * 30)

# Identity matrix
I = np.eye(3)
print(f"3×3 Identity matrix:")
print(I)

# Test identity property
A = np.random.randint(1, 10, (3, 3))
AI = A @ I
IA = I @ A
print(f"\nOriginal matrix A:")
print(A)
print(f"\nA @ I:")
print(AI)
print(f"\nI @ A:")
print(IA)
print(f"\nAre they equal? {np.allclose(A, AI) and np.allclose(A, IA)}")

# Diagonal matrix
diagonal_values = [2, 3, 5]
D = np.diag(diagonal_values)
print(f"\nDiagonal matrix with values {diagonal_values}:")
print(D)

# Symmetric matrix
# Create a symmetric matrix
B = np.random.rand(3, 3)
S = B + B.T  # Adding a matrix to its transpose makes it symmetric
print(f"\nSymmetric matrix S:")
print(S)
print(f"\nS transpose:")
print(S.T)
print(f"\nIs S symmetric? {np.allclose(S, S.T)}")

# Visualize these matrices
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
matrices = [I, D, S]
titles = ['Identity Matrix', 'Diagonal Matrix', 'Symmetric Matrix']

for i, (mat, title) in enumerate(zip(matrices, titles)):
    im = axes[i].imshow(mat, cmap='RdBu_r', aspect='equal')
    axes[i].set_title(title)
    
    # Add text annotations
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            axes[i].text(c, r, f'{mat[r,c]:.1f}', 
                        ha='center', va='center', 
                        color='white' if abs(mat[r,c]) > abs(mat).max()/2 else 'black')
    
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    plt.colorbar(im, ax=axes[i], shrink=0.6)

plt.tight_layout()
plt.show()
```

    Special Matrices:
    ==============================
    3×3 Identity matrix:
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    
    Original matrix A:
    [[8 5 4]
     [8 8 3]
     [6 5 2]]
    
    A @ I:
    [[8. 5. 4.]
     [8. 8. 3.]
     [6. 5. 2.]]
    
    I @ A:
    [[8. 5. 4.]
     [8. 8. 3.]
     [6. 5. 2.]]
    
    Are they equal? True
    
    Diagonal matrix with values [2, 3, 5]:
    [[2 0 0]
     [0 3 0]
     [0 0 5]]
    
    Symmetric matrix S:
    [[1.44399754 1.93076427 0.00784507]
     [1.93076427 1.23496302 0.63471559]
     [0.00784507 0.63471559 1.04954932]]
    
    S transpose:
    [[1.44399754 1.93076427 0.00784507]
     [1.93076427 1.23496302 0.63471559]
     [0.00784507 0.63471559 1.04954932]]
    
    Is S symmetric? True


    
![png](/materials/notebooks/02_matrices_and_operations/output_13_1.png)
    


## 7. Matrix Transpose

### Mathematical Definition
The transpose of matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ is $\mathbf{A}^T \in \mathbb{R}^{n \times m}$ where:
$$(\mathbf{A}^T)_{ij} = a_{ji}$$

### Properties
- $(\mathbf{A}^T)^T = \mathbf{A}$
- $(\mathbf{A} + \mathbf{B})^T = \mathbf{A}^T + \mathbf{B}^T$
- $(\mathbf{AB})^T = \mathbf{B}^T\mathbf{A}^T$


```python
# Matrix transpose
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print(f"Original matrix A (2×3):")
print(A)

# Transpose
A_T = A.T
print(f"\nTranspose A^T (3×2):")
print(A_T)

# Alternative method
A_T_alt = np.transpose(A)
print(f"\nUsing np.transpose (same result):")
print(A_T_alt)

# Verify properties
print(f"\nVerifying properties:")
print(f"(A^T)^T == A? {np.array_equal(A_T.T, A)}")

# For matrix multiplication property
B = np.array([[1, 2],
              [3, 4],
              [5, 6]])
AB = A @ B
AB_T = AB.T
BT_AT = B.T @ A.T

print(f"\nMatrix B (3×2):")
print(B)
print(f"\nAB (2×2):")
print(AB)
print(f"\n(AB)^T:")
print(AB_T)
print(f"\nB^T @ A^T:")
print(BT_AT)
print(f"\n(AB)^T == B^T A^T? {np.allclose(AB_T, BT_AT)}")

# Visualize transpose operation
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(A, cmap='viridis', aspect='equal')
plt.title('Original Matrix A')
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        plt.text(j, i, f'{A[i,j]}', ha='center', va='center', 
                color='white', fontsize=12, fontweight='bold')
plt.xticks([])
plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(A_T, cmap='viridis', aspect='equal')
plt.title('Transpose A^T')
for i in range(A_T.shape[0]):
    for j in range(A_T.shape[1]):
        plt.text(j, i, f'{A_T[i,j]}', ha='center', va='center', 
                color='white', fontsize=12, fontweight='bold')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()
```

    Original matrix A (2×3):
    [[1 2 3]
     [4 5 6]]
    
    Transpose A^T (3×2):
    [[1 4]
     [2 5]
     [3 6]]
    
    Using np.transpose (same result):
    [[1 4]
     [2 5]
     [3 6]]
    
    Verifying properties:
    (A^T)^T == A? True
    
    Matrix B (3×2):
    [[1 2]
     [3 4]
     [5 6]]
    
    AB (2×2):
    [[22 28]
     [49 64]]
    
    (AB)^T:
    [[22 49]
     [28 64]]
    
    B^T @ A^T:
    [[22 49]
     [28 64]]
    
    (AB)^T == B^T A^T? True


    
![png](/materials/notebooks/02_matrices_and_operations/output_15_1.png)
    


## 8. Matrix Properties and ML Applications

### Data Matrix Structure
In machine learning, we typically organize data as:
- **Rows**: Individual samples/observations
- **Columns**: Features/variables

$$\mathbf{X} = \begin{bmatrix} 
x_{11} & x_{12} & \cdots & x_{1n} \\
x_{21} & x_{22} & \cdots & x_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
x_{m1} & x_{m2} & \cdots & x_{mn}
\end{bmatrix}$$

where $m$ = number of samples, $n$ = number of features.


```python
# Practical ML example: Dataset matrix
print("Machine Learning Data Matrix Example:")
print("=" * 40)

# Create a sample dataset (housing prices)
# Features: [size (sq ft), bedrooms, age (years)]
# Samples: 5 houses
X = np.array([[1200, 3, 5],    # House 1
              [1500, 4, 10],   # House 2
              [800, 2, 2],     # House 3
              [2000, 5, 15],   # House 4
              [1000, 2, 8]])   # House 5

y = np.array([300000, 400000, 250000, 550000, 280000])  # Prices

print(f"Feature matrix X (5 samples × 3 features):")
print(X)
print(f"Shape: {X.shape}")
print(f"\nTarget vector y (5 prices):")
print(y)

# Access different parts of the data
print(f"\nFirst sample (house 1): {X[0, :]}")
print(f"Size feature for all houses: {X[:, 0]}")
print(f"Bedrooms feature for all houses: {X[:, 1]}")

# Add bias term (common in ML)
ones = np.ones((X.shape[0], 1))
X_with_bias = np.hstack([ones, X])  # Add column of 1s
print(f"\nX with bias term (5 × 4):")
print(X_with_bias)

# Linear regression setup: y = X @ theta
# We need to solve for theta (parameters)
theta = np.array([200, 10000, -5000, -2000])  # [bias, size_coef, bed_coef, age_coef]
print(f"\nParameter vector theta: {theta}")

# Predictions
y_pred = X_with_bias @ theta
print(f"\nPredicted prices: {y_pred}")
print(f"Actual prices:    {y}")
print(f"Errors:           {y - y_pred}")

# Visualize the data matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Heatmap of feature matrix
im1 = axes[0].imshow(X, cmap='viridis', aspect='auto')
axes[0].set_title('Feature Matrix X')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Samples (Houses)')
axes[0].set_xticks([0, 1, 2])
axes[0].set_xticklabels(['Size', 'Bedrooms', 'Age'])
axes[0].set_yticks(range(5))
axes[0].set_yticklabels([f'House {i+1}' for i in range(5)])

# Add text annotations
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        axes[0].text(j, i, f'{X[i,j]}', ha='center', va='center', 
                    color='white', fontweight='bold')

plt.colorbar(im1, ax=axes[0])

# Comparison of actual vs predicted prices
axes[1].scatter(y, y_pred, alpha=0.7, s=100)
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', alpha=0.8)
axes[1].set_xlabel('Actual Prices')
axes[1].set_ylabel('Predicted Prices')
axes[1].set_title('Actual vs Predicted Prices')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

    Machine Learning Data Matrix Example:
    ========================================
    Feature matrix X (5 samples × 3 features):
    [[1200    3    5]
     [1500    4   10]
     [ 800    2    2]
     [2000    5   15]
     [1000    2    8]]
    Shape: (5, 3)
    
    Target vector y (5 prices):
    [300000 400000 250000 550000 280000]
    
    First sample (house 1): [1200    3    5]
    Size feature for all houses: [1200 1500  800 2000 1000]
    Bedrooms feature for all houses: [3 4 2 5 2]
    
    X with bias term (5 × 4):
    [[1.0e+00 1.2e+03 3.0e+00 5.0e+00]
     [1.0e+00 1.5e+03 4.0e+00 1.0e+01]
     [1.0e+00 8.0e+02 2.0e+00 2.0e+00]
     [1.0e+00 2.0e+03 5.0e+00 1.5e+01]
     [1.0e+00 1.0e+03 2.0e+00 8.0e+00]]
    
    Parameter vector theta: [  200 10000 -5000 -2000]
    
    Predicted prices: [11975200. 14960200.  7986200. 19945200.  9974200.]
    Actual prices:    [300000 400000 250000 550000 280000]
    Errors:           [-11675200. -14560200.  -7736200. -19395200.  -9694200.]


    
![png](/materials/notebooks/02_matrices_and_operations/output_17_1.png)
    


## 9. Practice Exercises

Work through these exercises to solidify your understanding:


```python
# Practice Exercise: Matrix Operations
print("Practice Exercises:")
print("=" * 30)

# Create random matrices for practice
np.random.seed(123)  # For reproducible results
A = np.random.randint(1, 6, (3, 3))
B = np.random.randint(1, 6, (3, 3))
x = np.random.randint(1, 6, 3)

print(f"Matrix A:")
print(A)
print(f"\nMatrix B:")
print(B)
print(f"\nVector x: {x}")

# Exercise 1: Basic operations
print(f"\nExercise 1: Basic Operations")
print(f"A + B =")
print(A + B)

print(f"\nA - B =")
print(A - B)

print(f"\n2 * A =")
print(2 * A)

# Exercise 2: Matrix multiplication
print(f"\nExercise 2: Matrix Multiplication")
print(f"A @ B =")
print(A @ B)

print(f"\nB @ A =")
print(B @ A)
print(f"\nNote: A @ B ≠ B @ A (matrix multiplication is not commutative!)")

# Exercise 3: Matrix-vector multiplication
print(f"\nExercise 3: Matrix-Vector Multiplication")
y = A @ x
print(f"A @ x = {y}")

# Exercise 4: Transpose properties
print(f"\nExercise 4: Transpose Properties")
AT = A.T
BT = B.T
AB = A @ B
AB_T = AB.T
BT_AT = BT @ AT

print(f"(A @ B)^T =")
print(AB_T)
print(f"\nB^T @ A^T =")
print(BT_AT)
print(f"\nAre they equal? {np.allclose(AB_T, BT_AT)}")

# Exercise 5: Check if matrix is symmetric
print(f"\nExercise 5: Symmetry Check")
S = A @ A.T  # This creates a symmetric matrix
print(f"S = A @ A^T =")
print(S)
print(f"\nIs S symmetric? {np.allclose(S, S.T)}")
print(f"\nS^T =")
print(S.T)
```

    Practice Exercises:
    ==============================
    Matrix A:
    [[3 5 3]
     [2 4 3]
     [4 2 2]]
    
    Matrix B:
    [[1 2 2]
     [1 1 2]
     [4 5 1]]
    
    Vector x: [1 5 2]
    
    Exercise 1: Basic Operations
    A + B =
    [[4 7 5]
     [3 5 5]
     [8 7 3]]
    
    A - B =
    [[ 2  3  1]
     [ 1  3  1]
     [ 0 -3  1]]
    
    2 * A =
    [[ 6 10  6]
     [ 4  8  6]
     [ 8  4  4]]
    
    Exercise 2: Matrix Multiplication
    A @ B =
    [[20 26 19]
     [18 23 15]
     [14 20 14]]
    
    B @ A =
    [[15 17 13]
     [13 13 10]
     [26 42 29]]
    
    Note: A @ B ≠ B @ A (matrix multiplication is not commutative!)
    
    Exercise 3: Matrix-Vector Multiplication
    A @ x = [34 28 18]
    
    Exercise 4: Transpose Properties
    (A @ B)^T =
    [[20 18 14]
     [26 23 20]
     [19 15 14]]
    
    B^T @ A^T =
    [[20 18 14]
     [26 23 20]
     [19 15 14]]
    
    Are they equal? True
    
    Exercise 5: Symmetry Check
    S = A @ A^T =
    [[43 35 28]
     [35 29 22]
     [28 22 24]]
    
    Is S symmetric? True
    
    S^T =
    [[43 35 28]
     [35 29 22]
     [28 22 24]]


## Key Takeaways

1. **Matrices represent data**: In ML, matrices store datasets and transformations
2. **Matrix multiplication is fundamental**: Core operation in linear algebra and ML
3. **Dimensions must match**: Always check compatibility for operations
4. **Order matters**: Matrix multiplication is not commutative ($\mathbf{AB} \neq \mathbf{BA}$)
5. **Transpose has important properties**: Especially for matrix multiplication
6. **Special matrices have special roles**: Identity, diagonal, and symmetric matrices

## Next Steps

In the next notebook, we'll explore matrix inversion, determinants, and eigenvalues - concepts that are crucial for understanding how machine learning algorithms work under the hood.
