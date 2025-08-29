---
title: "Problem Set 0: Review and Chatting with LLMs"
layout: note
category: "Assignment"
permalink: /materials/assignments/hw0/
---


This is an individual assignment. Review assignment policy regarding collaboration and late submissions on website before starting.

Problem 1 should be submitted on Slack (see more in the instructions). Problem 2, 3, 4 and 5 should be submitted as a zip file, containing PDF files with names of the following format: `name_problem_i.pdf` for $i=3,4,5$, on moodle (submission link below). Problem 2 has a different name convention (see details in problem).

[Sumbission Link](https://lms.aub.edu.lb/mod/assign/view.php?id=2394563)

## Problem 1 (30 points): Data Collection and Analysis with Your Phone

This problem is designed to get you thinking creatively about how to collect and analyze data using limited (yet powerful) resources: your phone! You'll be surprised at how much data you can capture with your phone’s sensors. You might be less surprised at how much of that data is shared with tech companies, fueling predictions about your behavior, targeted ads, and addictive platform design.

Ethical dilemmas aside, this exercise is meant to show you what the **real-world iterative process of data science** looks like:  

collect data $\rightarrow$ define a problem $\rightarrow$ analyze it $\rightarrow$ redefine the problem $\rightarrow$ collect more data $\rightarrow$ try again.  

Not all data science is machine learning. Sometimes it's a matter of signal processing, statistics, or simple algorithms (e.g., Fourier transform, averages, locating objects in an image, tweaking thresholds, etc.).

You have to submit this problem as a *Colab notebook* link on the #phone-data-challenge channel on Slack. **Make sure that the link works before submitting, by making it publicly accessible.** Here's are more detailed instructions.

- **Use your phone as a sensor**: e.g. camera, microphone, gyroscope, accelerometer, light sensor, screen, etc. There are a bunch of apps that can give you access to these sensors (e.g. [Sensor Data Recorder](https://apps.apple.com/us/app/sensor-data-recorder/id1438400138))

- **Brainstorm 5 ideas**: Think of 5 possible uses of these sensors.  
  *Write them down at the top of the Jupyter notebook in a markdown cell.*

- **Choose one idea** and collect data for that idea.

- **Solve a task with the data — without machine learning**:  
  Examples (that you can't use):  
  - Find the location of an object of a certain color (e.g., red purse) in a photo or track it in a video.  
  - Detect the tempo of a song by shaking your phone in rhythm (using gyroscope/accelerometer data).
  - Record a sound (song, speech, etc.) with your microphone and analyze it.

- **You can use open-source libraries**, but not machine learning models.  
  (e.g., use OpenCV for color detection in images).

- **Any resource is allowed**. Disclose and briefly describe how you used LLMs by mentioning it in your notebook. 

- **Work individually**: *No teamwork for this problem.*

- **Submit your work**:  
  - Post a link to your Colab notebook in the `#problem-sets-subs` channel.  
  - Include a short description (max 3 sentences) on Slack explaining your idea.  
  - Your notebook should include your code, analysis, and any visualizations, and it should run.
  - The more you explain your process, the better. You can use the markdown cells to do that. 

- **Bonus / Penalty**:  
  - Bonus points for creativity and uniqueness.  
  - Penalty if your idea/solution is the same as someone else's.  
  - Part of your grade will be determined by your classmates. You can ``like'' (or add any emoji you want to) a submission. The most popular submissions ones get an extra bonus point. So the earlier you submit, the more points you can get!

---

## Problem 2 (20 points): Exploring Project Ideas with a Language Model

This assignment is a warm-up to get you thinking about **project ideas** and **datasets** before we dive into machine learning in detail. The best way to learn ML is through exposure to lots of examples. The more you explore, the better.  

Nowadays it is very easy to find interesting datasets, and brainstorm ideas with your favorite language model. Here's one way to start learning about what ML can do. 

(a) **Chat with an LLM (like ChatGPT, Gemini, Claude, etc.)**. No need to subscribe to any service. You can use the free versions. 
   - Pick a topic you're interested in (e.g. music, robotics, aerospace, cybersecurity, basketball, Formula 1...).  
   - Ask questions such as:  
     - *What are interesting applications of ML in X? Can you provide examples and references?*
     - *Give me original ideas for a project in X.*  
     - *Where could I get data for project X?*  
     - *If I were to take a course on ML in X, what would the syllabus look like? Can you make it more focused on theory/applications?*  

Be creative. The more specific and elaborate your questions, the more unique and interesting the answers will be.

(b) **Refine your exploration**  
   - Push the conversation deeper: ask about product ideas (if you're interested in applications/entrepreneurship) or paper ideas (if you're more research-oriented).  
   - Check what's already been done (Google, papers, projects) and include them in your project abstract submission (part c).

(c) **Create a project abstract**  
   - Ask the LLM to **draft an abstract** for a project proposal based on your discussion.  
   - This isn't your final project: it's just practice to see what's possible.

(d) **Create a Machine Learning for X syllabus**  
   - Ask the LLM to **draft a syllabus** for a course on ML for X. You can compare this syllabus with what you already know and the syllabus of this course.

(e) **Submit**: Submit PDF files of: 1) the abstract of the project idea (named `abstract.pdf`), 2) the syllabus of the course you've generated (named `syllabus.pdf`), and 3) the conversation history you've had with the LLM (named `chat_history.pdf`). You can use extensions like [this one](https://www.chatgpt2pdf.app/) to save PDF files. Include all the submissions in a final zip file of your submission on moodle. Feel 

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