---
title: "Problem Set 0: Data, Math, and Chatting with LLMs"
layout: note
category: "Assignment"
permalink: /materials/assignments/hw0/
---


This is an individual assignment. Review the [assignment policy](https://intro2ml.com/logistics/) regarding collaboration and late submissions on the website before starting.

Problem 1 should be submitted on Slack (see the instructions in the problem). Problem 2 is done entirely on the [course slides site](https://learn.sematlas.com/slides/ps0) and has nothing to upload. Problems 3, 4 and 5 are submitted as a zip file of PDFs named `name_problem_i.pdf` for $i=3,4,5$ on Moodle (link below).

[Sumbission Link](https://lms.aub.edu.lb/mod/assign/view.php?id=2820681)

## Problem 1 (30 points): Data Collection and Analysis with Your Phone

This problem is designed to get you thinking creatively about how to collect and analyze data using limited (yet powerful) resources: your phone! You'll be surprised at how much data you can capture with your phone’s sensors. You might be less surprised at how much of that data is shared with tech companies, fueling predictions about your behavior, targeted ads, and addictive platform design.

Ethical dilemmas aside, this exercise is meant to show you what the **real-world iterative process of data science** looks like:  

Here's the typical pipeline: Collect data, define a problem, analyze/visualize it, redefine the problem, collect more data, and repeat.  

Not all data science is machine learning. Sometimes it's a matter of signal processing, statistics, or simple algorithms (e.g., Fourier transform, averages, locating objects in an image, tweaking thresholds, etc.).

You have to submit this problem as a *Colab notebook* link on the `#ml-assignments` channel on Slack. **Make sure that the link works before submitting, by making it publicly accessible.** Here's are more detailed instructions.

- **Use your phone as a sensor**: e.g. camera, microphone, gyroscope, accelerometer, light sensor, screen, etc. There are a bunch of apps that can give you access to these sensors. The one I'm familiar with is: [Sensor Data Recorder](https://apps.apple.com/us/app/sensor-data-recorder/id1438400138), but you can use any app you want.

- **Brainstorm 5 ideas**: Think of 5 possible uses of these sensors. *Write them down at the top of the Jupyter notebook in a markdown cell.*
- **Choose one idea** and collect data for that idea.
- **Solve a task with the data — without machine learning**:  
  Examples (that you can't use):  
  - Find the location of an object of a certain color (e.g., red purse) in a photo or track it in a video.  
  - Detect the tempo of a song by shaking your phone in rhythm (using gyroscope/accelerometer data).
  - Record a sound (song, speech, etc.) with your microphone and analyze it.
- **You can use open-source libraries**, but not machine learning models. (e.g., use OpenCV for color detection in images).
- **Any resource is allowed**. Disclose and briefly describe how you used LLMs by mentioning it in your notebook. 
- **Submit your work**:  
  - Post a link to your Colab notebook in the `#ml-assignments` channel.  
  - Include a short description (max 3 sentences) on Slack explaining your idea.  
  - Your notebook should include your code, analysis, and any visualizations. Most importantly **it should run**.
  - The more you explain your process, the better. You can use the markdown cells to do that. 

- **Bonus / Penalty**:  
  - Bonus points for creativity and uniqueness.  
  - Penalty if your idea/solution is the same as someone else's.  
  - Part of your grade will be determined by your classmates. You can ``like'' (or add any emoji you want to) a submission. The most popular submissions ones get an extra bonus point. So the earlier you submit, the more points you can get!

---

## Problem 2 (20 points): Three Conversations

This problem is a warm-up: it gets you thinking about **project ideas** and
**datasets** before we go near the mathematics. The best way to learn ML early on
is exposure to lots of examples, so this is mostly exploration.

It also gets you set up on the course slides site, which is where every lecture
and every conversation lives this semester.

**Everything for this problem happens at
[learn.sematlas.com/slides/ps0](https://learn.sematlas.com/slides/ps0).** There
is nothing to submit on Moodle for Problem 2: no PDFs, no screenshots, no
browser extension to export a chat. Sign in with your AUB email and your real
name, and have the conversations there. I read them directly.

The deck opens with a short tour of the site, then three conversations:

**(a) What could ML do for X?** *(six turns minimum)* Pick a field you actually
care about, and work out what machine learning is really doing in it, what data
that requires, and what makes your problem hard for a machine specifically. You
should come out with three candidate project ideas.

**(b) Write the abstract.** *(five turns minimum)* Take the idea you liked best
and sharpen it into a project abstract: a named dataset or a real collection
procedure, a clear input and output, and a measure that would tell you whether
it worked. You write the abstract, not the model.

**(c) Design the course.** *(six turns minimum)* Build a syllabus for "Machine
Learning for X". Then compare it with the course you are sitting in, and tell me
one thing yours does better and one thing it does worse. I want the
disagreement.

A note on how these are graded. I am not grading the model's answers, and there
is no advantage in getting very long replies out of it. What
counts is how you push: whether you ask for specifics, whether you notice when an
answer is vague, whether you argue back.

Any trouble with the site, or anything that looks broken: press **C** on the
slide where it happened and it reaches me with the slide attached.

---

## Problem 3 (15 points): Gradients and Hessians

A matrix $ A \in \mathbb{R}^{n \times n} $ is symmetric if $ A^T = A $, that is $ A_{ij} = A_{ji} $ for all $ i, j $. Recall the gradient $ \nabla f(x) $ of a function $ f : \mathbb{R}^n \rightarrow \mathbb{R} $ which is the n-vector of partial derivatives:

$$ \nabla f(x) = \begin{bmatrix} \frac{\partial}{\partial x_1} f(x) \\ \vdots \\ \frac{\partial}{\partial x_n} f(x) \end{bmatrix} $$

where

$$ x = \begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix} $$

The Hessian $ \nabla^2 f(x) $ of a function $ f : \mathbb{R}^n \rightarrow \mathbb{R} $ is the $ n \times n $ symmetric matrix of twice partial derivatives:

$$ \nabla^2 f(x) = \begin{bmatrix} \frac{\partial^2}{\partial x_1^2} f(x) & \cdots & \frac{\partial^2}{\partial x_1 \partial x_n} f(x) \\ \vdots & \ddots & \vdots \\ \frac{\partial^2}{\partial x_n \partial x_1} f(x) & \cdots & \frac{\partial^2}{\partial x_n^2} f(x) \end{bmatrix} $$

(a) Let $ f(x) = \frac{1}{2} x^T Ax + b^T x $ where $ A $ is a symmetric matrix and $ b \in \mathbb{R}^n $ is a vector. What is $ \nabla f(x) $? Hint: spell-out the element-wise multiplication and deduce the expression from the resulting matrix. 

(b) Let $ f(x) = g(h(x)) $ where $ g : \mathbb{R} \rightarrow \mathbb{R} $ is differentiable and $ h : \mathbb{R}^n \rightarrow \mathbb{R} $ is differentiable. What is $ \nabla f(x) $?

(c) What is $ \nabla^2 f(x) $ for the $ f(x) $ from part (a)?

(d) [Extra credit] Let $ f(x) = g(a^T x) $ where $ g : \mathbb{R} \rightarrow \mathbb{R} $ is continuously differentiable and $ a \in \mathbb{R}^n $ is a vector. What are $ \nabla f(x) $ and $ \nabla^2 f(x) $? (Hint: your expression for $ \nabla^2 f(x) $ may have as few as 11 symbols including $ \nabla $ and parentheses.)

---

## Problem 4 (15 points): Positive Definite Matrices

A matrix $ A \in \mathbb{R}^{n \times n} $ is positive semi-definite (PSD), denoted $ A \succeq 0 $, if $ A = A^T $ and $ x^T Ax \geq 0 $ for all $ x \in \mathbb{R}^n $. A matrix $ A $ is positive definite, denoted $ A \succ 0 $, if $ A = A^T $ and $ x^T Ax > 0 $ for all non-zero $ x \in \mathbb{R}^n $.The simplest example of a positive definite matrix is the identity $ I $ (the diagonal matrix with 1s on the diagonal and 0s elsewhere), which satisfies $ x^T Ix = \|x\|^2 = \sum_{i=1}^n x_i^2 $.

(a) Let $ z \in \mathbb{R}^n $ be an n-vector. Show that $ A = zz^T $ is positive semidefinite.

(b) Let $ z \in \mathbb{R}^n $ be a non-zero n-vector. Let $ A = zz^T $. What is the null-space of $ A $? What is the rank of $ A $?

(c) Let $ A \in \mathbb{R}^{n \times n} $ be positive semidefinite and $ B \in \mathbb{R}^{m \times n} $ be arbitrary, where $ m, n \in \mathbb{N} $. Is $ BAB^T $ PSD? If so, prove it. If not, give a counterexample with explicit $ A, B $.

---

## Problem 5 (10 points): Eigenvectors, Eigenvalues, and the Spectral Theorem

The eigenvalues of an $ n \times n $ matrix $ A \in \mathbb{R}^{n \times n} $ are the roots of the characteristic polynomial $ p_A(\lambda) = \det(\lambda I - A) $, which may (in general) be complex. They are also defined as the values $ \lambda \in \mathbb{C} $ for which there exists a vector $ x \in \mathbb{C}^n $ such that $ Ax = \lambda x $. We call such a pair $ (x, \lambda) $ an eigenvector-eigenvalue pair. In this question, we use the notation $ \text{diag}(\lambda_1, ..., \lambda_n) $ to denote the diagonal matrix with diagonal entries $ \lambda_1, ..., \lambda_n $.


(a) Suppose that the matrix $ A \in \mathbb{R}^{n \times n} $ is diagonalizable, that is $ A = T \Lambda T^{-1} $ for an invertible matrix $ T \in \mathbb{R}^{n \times n} $ where $ \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n) $ is diagonal. Use the notation $ t^{(i)} $ for the columns of $ T $ so that $ T = [t^{(1)} \cdots t^{(n)}] $ where $ t^{(i)} \in \mathbb{R}^n $. Show that $ A t^{(i)} = \lambda_i t^{(i)} $ so that the eigenvalues/eigenvector pairs of $ A $ are $ (t^{(i)}, \lambda_i) $.

**Note:** A matrix $ U \in \mathbb{R}^{n \times n} $ is orthogonal if $ U^T U = I $. The spectral theorem, a crucial theorem in linear algebra, states that if $ A \in \mathbb{R}^{n \times n} $ is symmetric ($ A = A^T $), then $ A $ is diagonalizable by a real orthogonal matrix. In other words, there exists a diagonal matrix $ \Lambda \in \mathbb{R}^{n \times n} $ and an orthogonal matrix $ U \in \mathbb{R}^{n \times n} $ such that $ U^T A U = \Lambda $, or equivalently, 

$$ A = U \Lambda U^T $$

Let $ \lambda_i = \lambda_i(A) $ denote the $ i $th eigenvalue of $ A $.


(b) Let $ A $ be symmetric. Show that if $ U = [u^{(1)} \cdots  u^{(n)}] $ is orthogonal where $ u^{(i)} \in \mathbb{R}^n $ and $ A = U \Lambda U^T $ then $ u^{(i)} $ is an eigenvector of $ A $ and $ A u^{(i)} = \lambda_i u^{(i)} $ where $ \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n) $.

(c) [Extra Credit] Show that if $ A $ is PSD then $ \lambda_i(A) \geq 0 $ for each $ i $.

---