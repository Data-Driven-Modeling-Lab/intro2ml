---
title: "Problem Set 2: Logistic Regression and Classification"
layout: note
category: "Assignment"
permalink: /materials/assignments/hw2/
---


This is an individual assignment. I encourage you to discuss the problems with each other but the final write up has to be yours. Copying or sharing code is not allowed. The work you hand in has to be yours, and you have to be able to explain every line of it. Review the [assignment policy](https://intro2ml.com/logistics/) on collaboration and late submissions before you start.

Submission guidelines: Problems 0, 1, 2 and 3 are submitted as PDFs named `problem_x.pdf`. Problems 4 and 5 are submitted as Jupyter notebooks (`problem_4.ipynb` and `problem_5.ipynb`) that run top to bottom. Problems 6 and 7 are submitted on Slack and graded on your post and your comments (see the problems).

Deadlines: The homework is due on **Thursday 8 October at 23:59**. Zip what goes to Moodle into a single file and submit it here:
[Submission Link](https://lms.aub.edu.lb/mod/assign/view.php?id=2837312&forceview=1)

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

## Problem 2: The normal equation (10 points):

Given a linear model with fitting parameters $\theta$, linear hypothesis $h_\theta(x) = \theta^\top x$, and a dataset $\mathcal D = \{ (x^{(i)}, y^{(i)}) \}_{i=1}^{n}$, you can express the least squares cost in matrix form:

$$
J(\theta) = \lVert \mathbf X \theta - \mathbf y \rVert_2^2
$$

Derive the normal equation for the least squares problem. Here are some linear algebra identities that you might find useful:

$$
\nabla_{A^T} f(A) = (\nabla_A f(A))^T
$$

$$
\nabla_A \operatorname{tr}(ABA^TC) = CAB + C^TAB^T
$$

$$
\nabla_{A^T} \operatorname{tr}(ABA^TC) = BA^TC + B^TA^TC^T
$$

## Problem 3: Logistic regression derivations (20 points)

Given a dataset $\{ (x^{(i)}, y^{(i)}) \}_{i=1}^{n}$ with $x^{(i)} \in \mathbb R$ and $y^{(i)} \in \{0, 1\}$, we would like to fit a logistic classifier with fitting parameters $\theta$, and predictor 

$h_\theta(x) = g(\theta^\top \phi(x)) = \frac{1}{1 + e^{-\theta^\top \phi(x)}}$

(a) Find the derivative $g'(z) = dg/dz$ as a function of $g(z)$. Here $z \equiv \theta^\top \phi(x)$. 

(b) Write the likelihood $L(\theta) = \prod_i p(y^{(i)} \vert x^{(i)}; \theta)$ and the log-likelihood $\log L(\theta)$ in terms of $\theta$, $x^{(i)}$, and $y^{(i)}$.

(c) Derive the gradient of the log-likelihood $\nabla_\theta \log L(\theta)$ (you can use vector identities or take the derivative with respect to one parameter $\theta_j$ at a time, i.e. $\partial \log L(\theta) / \partial \theta_j$, where $\theta_j$ is the $j^{th}$ element of the vector $\theta$).

(d) Write the gradient ascent update rule that maximizes the log-likelihood, one example at a time. Compare it with the LMS update rule for linear regression.

## Problem 4: Classification with Scikit-Learn (30 points)

I have provided two files <a href="{{ 'materials/data/p3_x.txt' | relative_url }}" download>p3_x.txt</a> and <a href="{{ 'materials/data/p3_y.txt' | relative_url }}" download>p3_y.txt</a>. These files contain inputs $x^{(i)} \in \mathbb R^2$ and outputs $y^{(i)} \in \{ -1, 1 \}$, respectively, with one training example per row. This is a binary classification problem.

(a) Read the data (you can use [Pandas](https://pandas.pydata.org/)) from the files, and split it into training and test sets. Make sure to shuffle the data before splitting it.

(b) Plot the training data (your axes should be $x_1$ and $x_2$, corresponding to the two coordinates of the inputs, and you should use a different symbol for each point plotted to indicate whether that example had label 1 or -1, and whether it is a training or test data point). 

(c) Use [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) to fit a logistic regression model to the data. (Extra credit (5 points): use the stochastic gradient descent algorithm we wrote in class and make sure you get a similar result). 

(d) Plot the decision boundary. This should be a straight line showing the boundary separating the region where $h_\theta(\mathbf x) > 0.5$ from where $h_\theta(\mathbf x) \le 0.5$.). What is the test score of the model?

(e) What is the purpose of the penalty argument in the `LogisticRegression` classifier? Try the `L_1`, `L_2` and `ElasticNet` penalties and compare their decision boundaries as well as their test scores.

(f) How does [SVM](https://scikit-learn.org/stable/modules/svm.html) compare to Logistic Regression on this data-set? Briefly describe what loss function SVM is minimizing. Don't worry if you don't know what SVM does yet: the introduction on that scikit-learn page is enough for this question, and it connects to the margin we defined in class. You can simply pass the design matrix to the `SVC` class and set the `kernel` parameter to `linear`.

(g) Open ended, extra credit (5 points): search for 3 other classification algorithms that you can use on this data-set and state their advantages over logistic regression?

## Problem 5: Classification lab (30 points)

Complete the [Classification Metrics Lab notebook](https://intro2ml.com/materials/notebooks/classification_metrics_lab/). This is the in-class hands-on, now graded. Open it in Colab (or download it), fill in every cell marked `# TODO`, and answer the short questions in the markdown cells. You will load a real dataset (Breast Cancer Wisconsin), split it honestly with a stratified train and test split, fit logistic regression, and read the confusion matrix, precision, and recall rather than trusting accuracy; then move the decision threshold and trade precision against recall; and finally extend the same idea to many classes with softmax on the digits dataset.

Submit your completed notebook as `problem_5.ipynb`. It should run top to bottom.

## Problem 6: Project pre-proposal (10 points)

Before you form a team, propose your own project. The point is for everyone to think independently first, so that the teams you form later start from real ideas rather than convenience. Start by reading the project guidelines here: https://intro2ml.com/project/, then write a short pre-proposal (about 300 words) and post it on the #iml-project channel on Slack. Include:
- A title and a short description of the idea.
- The problem: what you would predict or model, and why it is interesting.
- The data: a dataset you could actually get, with a link if you have one. Finding a dataset first is usually easier than finding an idea first, so if you are stuck, start there.
- A hypothesis or question you would test, and one or two references or similar projects you would build on.
- A rough plan: the first method you would try, and how you would know whether it worked.

Before you post, run your draft through the pre-proposal reviewer on the slides: [PS2 on LearnSlides](https://learn.sematlas.com/slides/intro_ps2). Write the draft first, paste it in, and argue with the reviewer until the question, the data and the plan hold up; it will not write the draft for you. End your Slack post with one or two sentences on what the exchange changed your mind about. The conversation is saved and counts toward this problem.

Then read your classmates' posts and comment on at least two of them. A good comment is specific: ask a question about something that is unclear, point at a dataset or reference they might have missed, or propose an extension or a simpler baseline to start from. The quality of your two comments counts toward your grade for this problem.

## Problem 7: Engagement, and what it does to people (10 points)

You work at a social media company. The platform serves videos and short posts, like X or Instagram, and you are asked to build the machine learning system that maximizes engagement. Before you write any code, you have to decide what the problem even is.

(a) The technical setup. Write a concise definition, half a page at most, as `problem_7.pdf`:
- What is the problem setup? Is it supervised or unsupervised, or something else?
- What are the inputs and what is the output? What exactly do you fit?
- What does the data look like: one row of it, and where it comes from.
- What metric are you maximizing, and what does a good score mean for the company and for the user?
- There are many ways to define this problem. Name at least one other definition you rejected, and say why.

This is open ended. Go wild in how you define it, but define it precisely.

(b) The discussion. Summarize your definition in a short post on the `#discussion-forum` channel on Slack, where the PS1 discussion was, then discuss what it means. The theme of this semester is identity: how would a system like yours reshape the identities of the people who use it, and through which part of the machine learning is that happening? Consider in particular how your model's generalization, its overfitting, and the kind of data it is trained on would affect people's identities and self-image. Then read your classmates' posts and comment on at least two of them; specific, pointed comments count, agreement does not.
