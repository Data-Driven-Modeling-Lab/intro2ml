---
title: "Python Fundamentals for Machine Learning"
layout: note
category: "Jupyter Notebook"
permalink: /materials/notebooks/00_python_intro/
notebook_source: "00_python_intro.ipynb"
---


**A Quick Introduction/Refresher**

This notebook provides a rapid introduction to Python concepts essential for this course. We'll focus on the tools and syntax you'll use most frequently in machine learning.

## 1. Basic Python Syntax and Data Types

### Variables and Basic Operations


```python
# Variables - Python automatically determines the type
name = "Machine Learning"    # String
num_students = 30           # Integer
learning_rate = 0.01        # Float
is_supervised = True        # Boolean

print(f"Course: {name}")
print(f"Students: {num_students}")
print(f"Learning rate: {learning_rate}")
print(f"Supervised learning: {is_supervised}")

# Check types
print(f"\nType of name: {type(name)}")
print(f"Type of num_students: {type(num_students)}")
```

### Basic Math Operations


```python
# Basic arithmetic
a = 10
b = 3

print(f"Addition: {a} + {b} = {a + b}")
print(f"Subtraction: {a} - {b} = {a - b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division: {a} / {b} = {a / b}")
print(f"Integer division: {a} // {b} = {a // b}")
print(f"Modulo: {a} % {b} = {a % b}")
print(f"Exponentiation: {a} ** {b} = {a ** b}")

# Import math for more operations
import math
print(f"\nSquare root of {a}: {math.sqrt(a)}")
print(f"Log of {a}: {math.log(a)}")
```

## 2. Python Collections: Lists, Tuples, and Dictionaries

### Lists - Ordered, Mutable Collections


```python
# Create lists
features = ['age', 'income', 'education', 'experience']
scores = [85, 92, 78, 88, 95]
mixed_data = ["Alice", 25, True, 3.14]

print(f"Features: {features}")
print(f"Scores: {scores}")
print(f"Mixed data: {mixed_data}")

# List operations
print(f"\nFirst feature: {features[0]}")
print(f"Last score: {scores[-1]}")
print(f"First 3 scores: {scores[:3]}")
print(f"Length of features: {len(features)}")

# Modify lists
scores.append(90)  # Add element
print(f"After adding 90: {scores}")

features[0] = 'student_age'  # Modify element
print(f"After modifying first feature: {features}")
```

### List Comprehensions - Powerful and Pythonic


```python
# Traditional approach
squared_scores = []
for score in scores:
    squared_scores.append(score ** 2)
print(f"Squared scores (traditional): {squared_scores}")

# List comprehension - more concise
squared_scores_lc = [score ** 2 for score in scores]
print(f"Squared scores (list comp): {squared_scores_lc}")

# With condition
high_scores = [score for score in scores if score >= 90]
print(f"High scores (>= 90): {high_scores}")

# More complex example
normalized_scores = [(score - min(scores)) / (max(scores) - min(scores)) for score in scores]
print(f"Normalized scores: {[round(s, 3) for s in normalized_scores]}")
```

### Dictionaries - Key-Value Pairs


```python
# Create dictionaries
student = {
    'name': 'Alice',
    'age': 22,
    'major': 'Computer Science',
    'gpa': 3.85,
    'courses': ['Math', 'ML', 'Statistics']
}

print(f"Student info: {student}")
print(f"Name: {student['name']}")
print(f"GPA: {student['gpa']}")

# Dictionary operations
student['graduation_year'] = 2025  # Add new key
student['gpa'] = 3.90  # Update existing key

print(f"\nUpdated student: {student}")
print(f"Keys: {list(student.keys())}")
print(f"Values: {list(student.values())}")

# Iterate through dictionary
print("\nStudent details:")
for key, value in student.items():
    print(f"  {key}: {value}")
```

## 3. Control Flow: Loops and Conditionals

### If Statements


```python
def evaluate_performance(score):
    """Evaluate student performance based on score"""
    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Good"
    elif score >= 70:
        return "Satisfactory"
    elif score >= 60:
        return "Needs Improvement"
    else:
        return "Failing"

# Test the function
test_scores = [95, 85, 75, 65, 55]
for score in test_scores:
    performance = evaluate_performance(score)
    print(f"Score {score}: {performance}")
```

### Loops


```python
# For loops with range
print("Counting to 5:")
for i in range(5):
    print(f"  {i}")

print("\nEvery 2nd number from 0 to 10:")
for i in range(0, 11, 2):
    print(f"  {i}")

# Enumerate for index and value
algorithms = ['Linear Regression', 'Logistic Regression', 'Decision Trees', 'Neural Networks']
print("\nML Algorithms:")
for i, algorithm in enumerate(algorithms):
    print(f"  {i+1}. {algorithm}")

# While loop example
print("\nSimulating gradient descent iterations:")
loss = 10.0
iteration = 0
while loss > 0.1 and iteration < 10:
    loss *= 0.7  # Simulate decreasing loss
    iteration += 1
    print(f"  Iteration {iteration}: Loss = {loss:.3f}")
```

## 4. Functions - Building Reusable Code

### Basic Functions


```python
def calculate_mean(numbers):
    """Calculate the arithmetic mean of a list of numbers"""
    return sum(numbers) / len(numbers)

def calculate_variance(numbers):
    """Calculate the variance of a list of numbers"""
    mean = calculate_mean(numbers)
    return sum((x - mean) ** 2 for x in numbers) / len(numbers)

def calculate_std(numbers):
    """Calculate the standard deviation"""
    return math.sqrt(calculate_variance(numbers))

# Test our functions
data = [85, 92, 78, 88, 95, 82, 90, 87]
print(f"Data: {data}")
print(f"Mean: {calculate_mean(data):.2f}")
print(f"Variance: {calculate_variance(data):.2f}")
print(f"Standard Deviation: {calculate_std(data):.2f}")
```

### Functions with Default Parameters


```python
def linear_function(x, slope=1, intercept=0):
    """Compute y = slope * x + intercept"""
    return slope * x + intercept

def sigmoid(x, steepness=1):
    """Sigmoid activation function"""
    return 1 / (1 + math.exp(-steepness * x))

# Test functions
x_values = [-2, -1, 0, 1, 2]

print("Linear function (default parameters):")
for x in x_values:
    y = linear_function(x)
    print(f"  f({x}) = {y}")

print("\nLinear function (slope=2, intercept=3):")
for x in x_values:
    y = linear_function(x, slope=2, intercept=3)
    print(f"  f({x}) = {y}")

print("\nSigmoid function:")
for x in x_values:
    y = sigmoid(x)
    print(f"  sigmoid({x}) = {y:.3f}")
```

## 5. Essential Python Libraries for ML

### Importing Libraries


```python
# Different ways to import
import math                    # Import entire module
import numpy as np            # Import with alias (common convention)
from math import sqrt, log    # Import specific functions
import matplotlib.pyplot as plt  # Another common alias

# Check what we imported
print(f"π from math: {math.pi:.4f}")
print(f"e from numpy: {np.e:.4f}")
print(f"Square root of 16: {sqrt(16)}")
print(f"Natural log of e: {log(np.e):.4f}")
```

### Quick Preview of NumPy


```python
# NumPy arrays vs Python lists
python_list = [1, 2, 3, 4, 5]
numpy_array = np.array([1, 2, 3, 4, 5])

print(f"Python list: {python_list}")
print(f"NumPy array: {numpy_array}")
print(f"Type of list: {type(python_list)}")
print(f"Type of array: {type(numpy_array)}")

# Vector operations (much easier with NumPy!)
print("\nVector operations:")
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print(f"Array 1: {arr1}")
print(f"Array 2: {arr2}")
print(f"Element-wise addition: {arr1 + arr2}")
print(f"Element-wise multiplication: {arr1 * arr2}")
print(f"Dot product: {np.dot(arr1, arr2)}")
print(f"Squared: {arr1 ** 2}")
```

## 6. Error Handling and Debugging

### Common Error Types


```python
# Demonstrate common errors and how to handle them

def safe_divide(a, b):
    """Safely divide two numbers with error handling"""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print(f"Error: Cannot divide {a} by zero!")
        return None
    except TypeError:
        print(f"Error: Cannot divide {a} and {b} - check data types")
        return None

def safe_access_list(lst, index):
    """Safely access list element with error handling"""
    try:
        return lst[index]
    except IndexError:
        print(f"Error: Index {index} is out of range for list of length {len(lst)}")
        return None
    except TypeError:
        print(f"Error: Invalid index type {type(index)}")
        return None

# Test error handling
print("Testing safe_divide:")
print(f"10 / 2 = {safe_divide(10, 2)}")
print(f"10 / 0 = {safe_divide(10, 0)}")
print(f"10 / 'hello' = {safe_divide(10, 'hello')}")

print("\nTesting safe_access_list:")
test_list = [1, 2, 3, 4, 5]
print(f"list[2] = {safe_access_list(test_list, 2)}")
print(f"list[10] = {safe_access_list(test_list, 10)}")
print(f"list['hello'] = {safe_access_list(test_list, 'hello')}")
```

## 7. File I/O - Reading and Writing Data

### Basic File Operations


```python
# Writing data to a file
students_data = [
    "Alice,22,Computer Science,3.8",
    "Bob,21,Mathematics,3.6",
    "Charlie,23,Physics,3.9",
    "Diana,20,Chemistry,3.7"
]

# Write to file
with open('students.txt', 'w') as file:
    file.write("Name,Age,Major,GPA\n")  # Header
    for student in students_data:
        file.write(student + "\n")

print("Data written to students.txt")

# Read from file
with open('students.txt', 'r') as file:
    content = file.read()
    print("\nFile content:")
    print(content)

# Read line by line
print("Reading line by line:")
with open('students.txt', 'r') as file:
    for line_num, line in enumerate(file, 1):
        print(f"Line {line_num}: {line.strip()}")
```

### Working with CSV Data (Preview)


```python
import csv

# Read CSV data
def read_csv_data(filename):
    """Read CSV data and return as list of dictionaries"""
    data = []
    try:
        with open(filename, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                data.append(row)
        return data
    except FileNotFoundError:
        print(f"File {filename} not found")
        return []

# Read our student data
student_records = read_csv_data('students.txt')
print("Student records:")
for i, student in enumerate(student_records, 1):
    print(f"  {i}. {student}")

# Calculate average GPA
if student_records:
    gpas = [float(student['GPA']) for student in student_records]
    avg_gpa = sum(gpas) / len(gpas)
    print(f"\nAverage GPA: {avg_gpa:.2f}")

# Clean up
import os
if os.path.exists('students.txt'):
    os.remove('students.txt')
    print("\nCleaned up temporary file")
```

## 8. Quick Practice Exercises

Try these exercises to test your understanding!


```python
# Exercise 1: Create a function that normalizes a list of numbers to [0, 1]
def normalize_to_unit_range(numbers):
    """Normalize list of numbers to range [0, 1]"""
    if not numbers:
        return []
    
    min_val = min(numbers)
    max_val = max(numbers)
    
    if max_val == min_val:
        return [0.5] * len(numbers)  # All values are the same
    
    return [(x - min_val) / (max_val - min_val) for x in numbers]

# Test the function
test_data = [10, 20, 30, 40, 50]
normalized = normalize_to_unit_range(test_data)
print(f"Original: {test_data}")
print(f"Normalized: {normalized}")
```


```python
# Exercise 2: Create a simple grade book system
class GradeBook:
    def __init__(self):
        self.students = {}
    
    def add_student(self, name):
        if name not in self.students:
            self.students[name] = []
            print(f"Added student: {name}")
        else:
            print(f"Student {name} already exists")
    
    def add_grade(self, name, grade):
        if name in self.students:
            self.students[name].append(grade)
            print(f"Added grade {grade} for {name}")
        else:
            print(f"Student {name} not found")
    
    def get_average(self, name):
        if name in self.students and self.students[name]:
            return sum(self.students[name]) / len(self.students[name])
        return None
    
    def class_summary(self):
        print("\nClass Summary:")
        for name, grades in self.students.items():
            if grades:
                avg = self.get_average(name)
                print(f"  {name}: {grades} → Average: {avg:.1f}")
            else:
                print(f"  {name}: No grades yet")

# Test the grade book
gb = GradeBook()
gb.add_student("Alice")
gb.add_student("Bob")
gb.add_grade("Alice", 85)
gb.add_grade("Alice", 92)
gb.add_grade("Bob", 78)
gb.add_grade("Bob", 88)
gb.class_summary()
```

## Summary

You now have the Python fundamentals needed for this course! Key concepts covered:

1. **Basic syntax**: Variables, data types, operators
2. **Collections**: Lists, dictionaries, list comprehensions
3. **Control flow**: If statements, loops
4. **Functions**: Definition, parameters, return values
5. **Libraries**: Importing and using external code
6. **Error handling**: Try/except blocks
7. **File I/O**: Reading and writing data

## Next Steps

In the following notebooks, we'll dive deep into:
- **NumPy**: Efficient numerical computing
- **Matplotlib**: Data visualization
- **Pandas**: Data manipulation and analysis
- **Linear algebra**: The mathematical foundation of ML

These tools will be your foundation for everything we do in machine learning!
