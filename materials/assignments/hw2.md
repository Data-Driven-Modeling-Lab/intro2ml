---
title: "Problem Set 1: Linear Regression"
layout: note
category: "Assignment"
permalink: /materials/assignments/hw2/
---


This is an individual assignment. Review assignment policy regarding collaboration and late submissions on website before starting.

**Instructions:**: problem 4 will be evaluated based on your Slack participation. Problem 0, 1, 2 should be submitted as PDF, labeled as `problem_x.pdf`. Problem 3 should be submitted as a Jupyter notebook called `problem_3.ipynb` so that it can run locally. 

Zip all files and submit them as a single file on the following moodle link: [Sumbission Link](https://lms.aub.edu.lb/mod/assign/view.php?id=2408204)

## Problem 0: Ask a good question - LMLML (10 points)

In class, we showed that binary cross-entropy is theoretically better than ordinary least squares for classification with logistic regression. But how much better in practice? Explore this question with a language model. Design a simple experiment to test the theory by asking a language model to generate a synthetic dataset. Compare results using both loss functions. Optionally, run your own experiment on different datasets if the LLM doesn't yield the results you're looking for. Submit the PDF conversation (concatenated with a PDF of the Jupyter notebook of the experiments you've run yourself).

## Problem 1: Multiple Choice Questions (MCQ) Warm up (20 points)

Answer the following questions with True or False, and briefly explain your reasoning (2 sentences max): 

1. In logistic regression with a linear *logit*, the decision boundary is a hyperplane in the feature space. 
2. In logistic regression, the maximum likelihood estimator (MLE) can fail to exist (coefficients diverge to infinity) under perfect separation of the classes.
3. In logistic regression, using a threshold of 0.5 on the predicted probability yields a hard class label.
4. Using mean squared error (MSE) loss with a logistic (sigmoid) output guarantees a convex optimization problem. 
5. In a generalized linear model (GLM), the response distribution is assumed to be from the exponential family, and the mean is linked to a linear predictor via a link function. 
6. The canonical link in a GLM is always the identity link, regardless of the response distribution. 
7. Softmax (multinomial logistic) regression is *exactly* equivalent to training $K - 1$ independent one-vs-rest binary classifiers. 
8. Using cross-entropy loss with logistic/softmax outputs is standard because it matches the likelihood of the assumed distribution.
9. L2 regularization in logistic regression ensures that the model doesn't underfit on the training data.
10. Logistic regression assumes Gaussian white noise on the response variable $\theta^\top x$.

<!-- ## Problem 2: The normal equation (10 points):

Given a linear model with fitting parameters $\mathbf w$, linear hypothesis $h_\mathbf{w}(x) = \mathbf{w}^\top x$, and a dataset $\mathcal D = \{ (x^{(i)}, y^{(i)}) \}_{i=1}^{n}$, you can express the least mean square expression in matrix form: 

$ \mathcal L(\mathbf w) = \| \mathbf X \mathbf w - \mathbf y \|_2^2$, 

Derive the normal equation for the least squares problem. Here are some linear algebra identities that you might find useful:
$\nabla_{A^T} f(A) = (\nabla_A f(A))^T$
$\nabla_A tr(ABA^TC) = CAB + C^TAB^T$
$\nabla_{A^T} tr(ABA^TC) = BA^TC + B^TA^TC^T$ -->

## Problem 2: Logistic regression derivations (20 points)

Given a dataset $\{ (x^{(i)}, y^{(i)}) \}_{i=1}^{n}$ with $x^{(i)} \in \mathbb R$ and $y^{(i)} \in \mathbb [0, 1]$, we would like to fit a logistic classifier with fitting parameters $\mathbf w$, and predictor 

$f_\mathbf{w}(x) = g(\phi(x) \cdot \mathbf w) = \frac{1}{1 + e^{-\phi(x) \cdot \mathbf w}}$

(a) Find the derivative $g'(z) = dg/dz$ as a function of $g(z)$. Here $z \equiv \phi(x) \cdot \mathbf w$. 

(b) Find the log-likelihood $l(\mathbf w)$ from the likelihood $p(y^{(i)} \vert x^{(i)}; \mathbf w)$ in terms of $\mathbf w$, $x^{(i)}$, and $y^{(i)}$.

Derive the equation for the gradient of the log-likelihood $\nabla_\mathbf{w} l(\mathbf w)$ (you can use vector identities or get the derivative with respect to one parameter ($w_j$) at a time, i.e. $\partial l(\mathbf w) / \partial w_j$, where $w_j$ is the $j^{th}$ element of the vector $\mathbf w$.

(c) What is the least mean squares (LMS) update rule to maximize the log-likelihood?

## Problem 3: Classification with Scikit-Learn (30 points)

I have provided two files <a href="{{ 'materials/data/p2_x.txt' | relative_url }}" download>p2_x.txt</a> and <a href="{{ 'materials/data/p2_y.txt' | relative_url }}" download>p2_y.txt</a>. These files contain inputs $x^{(i)} \in \mathbb R^2$ and outputs $y^{(i)} \in \{ -1, 1 \}$, respectively, with one training example per row. This is a binary classification problem.

(a) Read the data (you can use [Pandas](https://pandas.pydata.org/)) from the files, and split it into training and test sets. Make sure to shuffle the data before splitting it.

(b) Plot the training data (your axes should be $x_1$ and $x_2$, corresponding to the two coordinates of the inputs, and you should use a different symbol for each point plotted to indicate whether that example had label 1 or -1, and whether it is a training or test data point). 

(c) Use [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) to fit a logistic regression model to the data. (Extra credit (5 points): use the stochastic gradient descent algorithm we wrote in class and make sure you get a similar result). 

(d) Plot the decision boundary. This should be a straight line showing the boundary separating the region where $f_\mathbf{w}(\mathbf x) > 0.5$ from where $f_\mathbf{w}(\mathbf x) \le 0.5$.). What is the test score of the model?

(e) What is the purpose of the penalty argument in the `LogisticRegression` classifier? Try the `L_1`, `L_2` and `ElasticNet` penalties and compare their decision boundaries as well as their test scores.

(f) How does [SVM](https://scikit-learn.org/stable/modules/svm.html) compare to Logistic Regression on this data-set? Briefly describe what loss function SVM is minimizing. Don't worry if you don't know what SVM does yet (we'll cover this in the coming week). You can simply pass the design matrix to the `SVC` class and set the `kernel` parameter to `linear`.

(g) Open ended, extra credit (5 points): search for 3 other classification algorithms that you can use on this data-set and state their advantages over logistic regression?

## Problem 4: Project pre-proposal (10 points)

This week, you're required to finalize your project team, and submit a short description of your proposed project by next Monday. The proposal will help us guide you towards a fruitful project. Start by reading the project guidelines here: https://intro2ml.com/project/

Discuss a project idea with your teammates and submit a short description on the #project-hub Slack channel. Include the following:
- A title and short description of the project idea (max 200 words)
- A list of team members tagged with the @ symbol
- A link to similar projects that you aim to build on (based on a literature review of similar ideas)
- Optionally: references to datasets that you aim to use in your project

This will push you to start thinking seriously about the scope of your project, and to find a dataset that you can use. I recommend starting from finding a dataset: it's easier to develop models given a dataset than to find a dataset given idea. We'll give you feedback and tips for your idea once you submit it on Slack.