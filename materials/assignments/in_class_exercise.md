# In-Class Group Exercise: Data Exploration and Analysis

**Time:** 25 minutes  
**Group Size:** 2-3 students  
**Dataset:** `student_survey_data.csv`

## Objective
Work in groups to explore a real dataset, understand its structure, and answer specific questions using Python data analysis and visualization techniques.

## Setup (5 minutes)

### Step 1: Download the Data
- Download `student_survey_data.csv` from the course materials
- Place it in the same directory as your Jupyter notebook

### Step 2: Form Groups
- Work in pairs or groups of 3
- Make sure at least one person in each group has Python/Jupyter running

### Step 3: Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 6)
```

## Part 1: Data Discovery (8 minutes)

### Task 1.1: Load and Examine the Data
```python
# Load the dataset
df = pd.read_csv('student_survey_data.csv')

# Your code here - explore the basic structure
```

**Questions to answer:**
1. How many students are in the dataset?
2. How many variables (columns) are there?
3. What are the column names?
4. What types of data do we have (numerical vs categorical)?

### Task 1.2: Basic Statistics
**Questions to investigate:**
1. What's the average GPA in the dataset?
2. What's the range of study hours per week?
3. How many students are in each major?
4. What's the most common year (Freshman, Sophomore, etc.)?

**Group Discussion Point:** What surprises you about this data? What patterns do you notice immediately?

## Part 2: Data Analysis (10 minutes)

### Task 2.1: Explore Relationships
Pick **TWO** of these research questions to investigate:

**Option A: Study Habits and Performance**
- Is there a relationship between study hours per week and GPA?
- Create a scatter plot and calculate the correlation
- Do different majors have different study patterns?

**Option B: Lifestyle and Academic Performance**
- How do sleep hours relate to GPA?
- What about coffee consumption and study hours?
- Create visualizations to explore these relationships

**Option C: Stress and Balance**
- What factors are related to stress level?
- How do exercise hours, social media usage, and study hours relate to stress?
- Create plots to investigate these relationships

**Option D: Major Differences**
- Do different majors have different average GPAs?
- What about programming experience by major?
- Create comparison plots (box plots or bar charts)

### Task 2.2: Create Your Best Visualization
Create **one compelling visualization** that tells an interesting story about the data. Consider:
- What relationship or pattern is most interesting?
- How can you make the plot clear and informative?
- What title and labels make it easy to understand?

## Part 3: Group Presentations (2 minutes each group)

Each group will briefly present (2 minutes max):
1. **One surprising finding** from your data exploration
2. **Your best visualization** - show the plot and explain what it reveals
3. **One question** the data raises that you'd like to investigate further

### Presentation Template:
- "We found that..."
- "This plot shows..."
- "One question we'd like to explore further is..."

## Helpful Code Snippets

### Basic Data Exploration
```python
# Dataset overview
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Summary statistics
df.describe()

# Value counts for categorical variables
df['major'].value_counts()
```

### Creating Plots
```python
# Scatter plot
plt.scatter(df['study_hours_per_week'], df['gpa'])
plt.xlabel('Study Hours per Week')
plt.ylabel('GPA')
plt.title('Study Hours vs GPA')
plt.show()

# Box plot by category
sns.boxplot(data=df, x='major', y='gpa')
plt.xticks(rotation=45)
plt.show()

# Correlation
correlation = df['study_hours_per_week'].corr(df['gpa'])
print(f"Correlation: {correlation:.3f}")
```

### Grouping and Aggregation
```python
# Average GPA by major
df.groupby('major')['gpa'].mean()

# Multiple statistics
df.groupby('major').agg({
    'gpa': 'mean',
    'study_hours_per_week': 'mean',
    'stress_level': 'mean'
}).round(2)
```

## Assessment Criteria

You'll be assessed on:
- **Collaboration**: Working effectively as a team
- **Exploration**: Asking good questions and investigating the data systematically
- **Visualization**: Creating clear, informative plots
- **Communication**: Presenting findings clearly and concisely
- **Critical Thinking**: Drawing reasonable conclusions from the data

## Bonus Challenges (If You Finish Early)

1. **Advanced Visualization**: Create a multi-panel figure showing several relationships
2. **Statistical Testing**: Use correlation tests or t-tests to check if relationships are statistically significant
3. **Data Quality**: Check for outliers or unusual patterns in the data
4. **Feature Engineering**: Create new variables (e.g., total screen time = social media + study hours)

## Common Issues and Solutions

### Problem: "FileNotFoundError"
**Solution**: Make sure the CSV file is in the same directory as your notebook

### Problem: "Module not found"
**Solution**: Install missing packages: `pip install pandas matplotlib seaborn`

### Problem: Plot not showing
**Solution**: Add `plt.show()` after creating plots

### Problem: Categorical data not plotting well
**Solution**: Try using `sns.boxplot()` or `sns.countplot()` instead of scatter plots

## Key Learning Objectives

By the end of this exercise, you should be able to:
 Load data from a CSV file using pandas  
 Explore data structure and basic statistics  
 Create meaningful visualizations  
 Calculate correlations and group statistics  
 Work collaboratively on data analysis  
 Present findings clearly and concisely  

## Reflection Questions (For After the Exercise)

1. What was the most challenging part of this exercise?
2. What did you learn about the relationship between different variables?
3. If you had more time, what would you investigate next?
4. How might this type of analysis be useful in machine learning?

---

**Remember**: The goal is to explore, learn, and have fun with real data! Don't worry about finding the "perfect" answer - focus on the process of investigation and discovery.