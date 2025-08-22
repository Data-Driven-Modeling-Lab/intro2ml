---
title: "Define the polynomial model"
layout: note
category: "Reference Material"
permalink: /materials/references/kaggle_competition/
notebook_source: "kaggle_competition.ipynb"
---

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the polynomial model
def polynomial_model(x1, x2):
    return (
        7.2 * x1 +
        4.333 * x2 +
        -8.8 * x1 * x2 +
        -2.555 * x2**2 +
        -1.12 * x1**2 * x2
    )

# x1_values = np.linspace(-10, 10, 100)
# x2_values = np.linspace(-10, 10, 100)
# Generate random data in the range (-10, 10)
x1_values = np.random.uniform(-10, 10, 100)
x2_values = np.random.uniform(-10, 10, 100)
x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)

# Flatten the grid to create a full dataset
x1_flat = x1_grid.flatten()
x2_flat = x2_grid.flatten()
y_values = polynomial_model(x1_flat, x2_flat)

# Create a DataFrame for the full dataset
data = pd.DataFrame({'x1': x1_flat, 'x2': x2_flat, 'y': y_values})

# Add noise to the dataset
noise = np.random.normal(0, np.sqrt(80), size=len(data))
data['y_noisy'] = data['y'] + noise

# Define regions to remove (as test data)
def remove_large_patches(data, patches):
    test_data = pd.DataFrame(columns=['x1', 'x2', 'y_noisy'])
    for patch in patches:
        x1_min, x1_max, x2_min, x2_max = patch
        removed_data = data[(data['x1'].between(x1_min, x1_max)) & (data['x2'].between(x2_min, x2_max))] 
        data = data[~((data['x1'].between(x1_min, x1_max)) & (data['x2'].between(x2_min, x2_max)))]
        test_data = pd.concat([test_data, removed_data]) 
    return data, test_data 

# Define larger patches to remove (for testing)
large_patches = [
    (-10, -5, -10, -5),  # Bottom-left corner
    (5, 10, 5, 10),      # Top-right corner
    (-10, -5, 5, 10),    # Top-left corner
    (5, 10, -10, -5),    # Bottom-right corner
    (-4, 4, -4, 4)       # Large patch in the center
]

# Remove larger patches and split into train and test
train_data_sample, test_data_sample = remove_large_patches(data.copy(), large_patches)

# Create true test dataset without noise for ground truth
true_test_data = test_data_sample[['x1', 'x2', 'y_noisy']]

# Create sample submission with placeholder predictions
sample_submission = pd.DataFrame({
    'id': range(len(test_data_sample)),
    'y': [0.0] * len(test_data_sample)  # Placeholder values for predictions
})

# Create the true test solution with the usage column (half public, half private)
true_test_data['id'] = range(len(true_test_data))
true_test_data['Usage'] = ['Public'] * (len(true_test_data) // 2) + ['Private'] * (len(true_test_data) - len(true_test_data) // 2)

# # Save the datasets to CSV
# train_data_sample[['x1', 'x2', 'y_noisy']].to_csv('train.csv', index=False)
# test_data_sample[['x1', 'x2']].to_csv('test.csv', index=False)
# sample_submission.to_csv('sample_submission.csv', index=False)
# true_test_data[['id', 'y_noisy', 'Usage']].to_csv('true_test.csv', index=False)

```


```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib widget

# Function to plot training and test data
def plot_datasets(train_data, test_data):
    fig = plt.figure(figsize=(14, 8))
    
    # Plotting the training data
    ax_train = fig.add_subplot(121, projection='3d')
    ax_train.scatter(train_data['x1'], train_data['x2'], train_data['y_noisy'], c=train_data['y_noisy'], cmap='viridis', s=14)
    ax_train.set_title("Training Data (with Noise)")
    ax_train.set_xlabel('x1')
    ax_train.set_ylabel('x2')
    ax_train.set_zlabel('y_noisy')
    ax_train.grid(False)
    
    # Plotting the test data
    ax_test = fig.add_subplot(122, projection='3d')
    ax_test.scatter(test_data['x1'], test_data['x2'], test_data['y_noisy'], c=test_data['y_noisy'], cmap='plasma', s=14)
    ax_test.set_title("Test Data (Ground Truth)")
    ax_test.set_xlabel('x1')
    ax_test.set_ylabel('x2')
    ax_test.set_zlabel('y (Ground Truth)')
    
    plt.tight_layout()
    plt.show()

# Plot the datasets
plot_datasets(train_data_sample, true_test_data)
```


    ![png](/materials/notebooks/kaggle_competition/output_1_0.png)

```python
# define and run linear regression on training data

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define polynomial features
def poly_features(data, degree=1):
    X = data[['x1', 'x2']]
    for d in range(2, degree + 1):
        for i in range(d + 1):
            for j in range(i + 1):
                X[f'x1^{i-j} * x2^{j}'] = data['x1']**(i-j) * data['x2']**j
    return X

degree = 7
X_train = poly_features(train_data_sample, degree=degree)
y_train = train_data_sample['y_noisy']

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
X_test = poly_features(true_test_data[['x1', 'x2']], degree=degree)
true_test_data['y_pred'] = model.predict(X_test)
train_data_sample['y_pred'] = model.predict(X_train)

# Calculate the mean squared error
mse = mean_squared_error(true_test_data['y_noisy'], true_test_data['y_pred'])
mse_train = mean_squared_error(train_data_sample['y_noisy'], train_data_sample['y_pred'])
print(f"Mean Squared Error: {mse, mse_train}")

# Plot the true test data with predictions
def plot_predictions(test_data):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(test_data['x1'], test_data['x2'], test_data['y_pred'], 'b.')
    ax.plot(test_data['x1'], test_data['x2'], test_data['y_noisy'], 'r.')
    ax.set_title("True Test Data with Predictions")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.tight_layout()
    plt.show()

plot_predictions(true_test_data)
print(model.coef_)
```

    Mean Squared Error: (287.42170568440343, 81.03980653746657)


    /var/folders/q4/_twpfpf54f3f6s17s74p67tc0000gp/T/ipykernel_36754/3631114194.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X[f'x1^{i-j} * x2^{j}'] = data['x1']**(i-j) * data['x2']**j


    ![png](/materials/notebooks/kaggle_competition/output_2_1.png)

    [-2.35118395e+07  2.33694246e+00  5.25058640e-04  2.35118469e+07
      2.33409964e+00 -2.01114287e-01 -8.88286407e+00 -2.74424024e+00
     -2.27805450e-02 -1.16091369e+00 -4.45679027e-02 -1.30260929e-02
      2.87672775e-03  3.01417946e-03  7.20338696e-03  2.21456889e-03
      2.65023424e-03  3.90356242e-04  8.62030577e-04  1.80951518e-03
      1.39642965e-03  8.59116598e-04  6.68692876e-05 -1.14436869e-05
     -2.67816743e-05 -5.36327920e-05 -2.51672228e-05 -4.52221818e-05
     -1.51830784e-05 -1.20228811e-05 -1.90835563e-06 -4.90950531e-06
     -1.35155894e-05 -1.31903029e-05 -1.35162111e-05 -9.28410861e-06
     -4.60870626e-06  3.62656779e-07]


```python
# save predictions

true_test_data[['id', 'y_pred']].to_csv('true_test_predictions.csv', index=False)
```


```python

```
