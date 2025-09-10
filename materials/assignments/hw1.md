---
title: "Problem Set 1: Linear Regression"
layout: note
category: "Assignment"
permalink: /materials/assignments/hw1/
---


This is an individual assignment. Review assignment policy regarding collaboration and late submissions on website before starting.

**Instructions:**: problem 0 will be evaluated based on your Slack participation. Problem 1 and 5 should be submitted as PDF, labeled as `problem_1.pdf` and `problem_5.pdf` respectively. Problem 2 should be submitted as a Jupyter notebook called `problem_2.ipynb` so that it can run locally. 

Zip all files and submit them as a single file on the following moodle link: [Sumbission Link](https://lms.aub.edu.lb/mod/assign/view.php?id=2401976)

## Problem 0: Ask a good question (10 points)

To encourage participation and interactions on Slack, you are required to ask at least one good question about the lectures we covered so far, and answer at least one question from another student. Do that on the #learn-and-share channel.

## Problem 1: Multiple Choice Questions (MCQ) Warm up (20 points)

Answer the following questions with True or False, and briefly explain your reasoning (2 sentences max): 

1.  Ordinary Least Squares (OLS) minimizes the sum of squared residuals between predictions and targets. 
2.  Adding a feature that is a perfect linear combination of existing features can make the OLS solution non-unique. 
3.  Normalizing features changes the location of the OLS minimum in parameter space. 
4.  In polynomial regression, increasing the degree always reduces training error but may increase test error. 
5.  L2 regularization (ridge) tends to set some coefficients exactly to zero.
6.  In k-fold cross-validation, using the same random seed across runs can change the average validation score. 
7.  Early stopping acts like implicit regularization by limiting effective model capacity. 
8.  When features are highly correlated, ridge typically outperforms OLS on test error due to variance reduction.
9.  Stochastic Gradient Descent (SGD) with a fixed learning rate will always converge if we run it long enough. 
10. The purpose of a test set is to prevent overfitting.


## Problem 2: Work with this dataset (20 points)

Apply linear regression to the following [dataset](https://www.opendatalebanon.org/job/weather/). This dataset contains both monthly and yearly weather statistics from 1996 to 2018. This dataset is aggregated by source (Rafiq Al-Hariri Int. airport, Beirut golf, Zahle, Tripoli), temperature (minimum, maximum, average), rain and humidity (humidity rates, maximal wind direction, wind power, rainy days, mm of rain, etc).

Create a Jupyter notebook that loads the dataset, preprocesses it, and applies linear regression to it. Here are more detailed instructions:
- Load the dataset 
- Preprocess the dataset to make sure you account for missing values
- Visualize the dataset to understand the structure
- Explore the dataset and decide which measurements are relevant for the problem at hand (monthly/yearly, temperature, rain, humidity, etc.)
- Define the features and target variable
- Apply linear regression to the dataset and analyze the results/plots
- Suggest improvements to the model and explain what experiments you would do to improve the model
- Propose real-world scenarios where this model could be useful

Submit the notebook with the dataset in the same folder called `problem_2.ipynb` so that it can run locally. If it doesn't run, you will lose points.


## Problem 3: Linear Regression and Gradient Descent (20 points)

Consider a supervised learning problem with inputs $x\in\mathbb{R}^d$ and targets $y\in\mathbb{R}$. We use a linear hypothesis with a bias term via the "bias trick": let $x_0=1$ and define

$$
h_\theta(x)=\theta^\top x,\qquad \theta\in\mathbb{R}^{d+1},\; x\in\mathbb{R}^{d+1}.
$$

We fit $\theta$ by minimizing the **squared-loss** cost

$$
J(\theta)\;=\;\tfrac12\sum_{i=1}^n\big(h_\theta(x_i)-y_i\big)^2.
$$

(a) Explain why adding a constant feature $x_0=1$ (the “bias trick”) is equivalent to allowing the fitted hyperplane to *not* pass through the origin. Describe this geometrically in $\mathbb{R}^2$ (one real feature plus bias).

(b) Briefly argue why minimizing the residual (no absolute value or square) is a bad idea. Then compare the absolute loss to the squared loss in terms of differentiability and suitability for gradient-based optimization.

(c) Show that $J(\theta)$ is a convex function of $\theta$. Under what condition on the data matrix $X$ (whose rows are $x_i^\top$) is $J$ **strictly** convex, hence the minimizer unique?

(d) Derive $\nabla_\theta J(\theta)$. Then write one step of **batch** gradient descent (GD) with learning rate $\alpha>0$:

$$
\theta \leftarrow \theta - \alpha\,\nabla_\theta J(\theta).
$$

(You may present the gradient either as a sum over examples or in matrix form as $X^\top(X\theta-y)$, where you need to define $X$ and $y$ properly.)

(e) Use the dataset with two examples (already bias-augmented):

$$
(x_1,y_1)=\big([1,\,2],\,5\big),\qquad (x_2,y_2)=\big([1,\,-1],\,0\big).
$$

Start from $\theta^{(0)}=(0,\,0)$ and take **one** batch GD step with $\alpha=0.1$. Compute $\theta^{(1)}$.

(f) From the same start $\theta^{(0)}=(0,\,0)$ and $\alpha=0.1$, perform **stochastic** GD visiting $x_1$ first, then $x_2$ (one pass over the two points). Report $\theta$ after both updates. Briefly comment on how this path can differ from batch GD even though both descend $J$.

(g) Describe the shape of the level sets (contours) of $J(\theta)$ for two parameters $(\theta_0,\theta_1)$ and how the gradient direction relates to those contours. Explain qualitatively what can happen if $\alpha$ is chosen too large vs. reasonably small.

## Problem 4: Kaggle Competition (20 points)

Sign up to kaggle.com and make teams of up to 3. And join the competition through this link: [Competition Link](https://www.kaggle.com/competitions/find-polynomial). Please don’t share the link with people outside the class (for now). The problem statement is to fit a function given two inputs and one output (all real numbers). You’re free to use whatever method you see fit. You’re allowed only 2 submissions per day. You'll see more details on the link.

## Problem 5: Learning Machine Learning with Machine Learning - LMLML (20 points)

* Explaining core ideas in your own words (features, model/hypothesis, loss, optimization).
* Doing math: gradients, normal equation, regularization effects.
* Coding + plotting a simple regression experiment.
* Testing yourself with MCQs and reflecting on mistakes.

#### Deliverables (submit as **one PDF**)

1. **Full chat transcript** with the model (from your first prompt to the end).
2. **MCQ answer sheet** (Q1–Q10 with your chosen option + brief reason).
3. **Reflection (<300 words)** using *What? So what? Now what?*
4. **Code & plots** shown in the conversation. 
5. **Link(s) to the lecture slides/video** you used (if any).


You will run **one** sustained conversation with ChatGPT (or a similar LLM). Start with the **Initial Prompt** below by copy-pasting it into the chat as your very first message.

#### Initial Prompt (copy-paste this)
```
You are an AI tutor helping me master *linear regression*. This week I learned about:
- Linear regression hypothesis and model
- feature engineering,
- mean squared error and gradients,
- gradient descent vs. stochastic gradient descent,
- step size/learning rate schedules,
- train/val/test splits and generalization,
- L2 regularization (ridge),
- normal equation.

Your role and ground rules:
1) Start by asking me a brief calibration question about my background (math & Python).
2) Then immediately give me **5 multiple-choice questions** (MCQs) that mix:
   - concepts (assumptions, scaling effects),
   - lightweight math (one gradient/normal-equation item),
   - a tiny numeric step of gradient descent,
   - a coding/output interpretation item,
   - one regularization question.
   For each MCQ: wait for my answer before revealing the solution; after I answer, give me a concise explanation.
3) Throughout, whenever I ask "show me the math," include the derivation using clear steps and notation.
4) When I ask for code or a plot, give runnable Python (NumPy/Matplotlib) with a tiny synthetic dataset so I can run it locally; label axes and include comments. Whenever possible, run it on your side and show me the plots.
5) Help me *push deeper*. Offer optional extensions like:
   - comparing GD vs. SGD on a noisy dataset,
   - effect of $\lambda$ in ridge (bias–variance),
   - learning-rate schedules or early stopping.
6) By the end, after I request it, provide more MCQs at a slightly higher difficulty, again revealing explanations only after I answer.
7) If I seem confused, use short Socratic questions to guide me.
8) Keep answers crisp, math correct, and code minimal but complete.
9) At the very end, summarize the topics I've learned and show me all the MCQs I answered, which ones I got correctly, and key takeaways.

Assume I may share a link to my week's lecture; if I do, you can tailor the depth. Ready? Ask your calibration question first.
```

#### Required elements during your chat

* Answer the **first 5 MCQs** one by one (explanations revealed after each).
* Ask the tutor for **“5 more MCQs”** near the end, then answer them (total = **10 MCQs**).
* Ask for **at least one derivation** (e.g., gradient of MSE, normal equation, or ridge closed-form).
* Ask for **at least one code snippet + plot** exploring linear regression (e.g., fit vs. no scaling, $\lambda$ sweep for ridge, GD vs. SGD trajectories).
* Ask **at least two off-script questions** that go beyond lecture scope (examples below).
* Request a **final summary** and **two specific next steps** from the tutor at the end.


#### Suggested conversation flow 

1. Paste the **Initial Prompt**.
2. Answer the calibration question.
3. Do the **first 5 MCQs**.
4. Depending on your MCQs, e.g. ask: "Show me the math for the gradient of the MSE loss and one GD update step.", "Give me a tiny experiment sweeping $\lambda$ in ridge regression (e.g., $\lambda \in {0, 0.1, 1, 10})$; plot train/test MSE.", etc.
5. Dig deeper with more MCQs: e.g. "Give me one more difficult MCQ question about the logic behind least squares".
6. Ask more questions until you have answered at least 10 MCQs.
7. Ask: "Rewrite a review of all MCQs, and my answers with a brief explanation."

Off-script prompts you *might* try (please tailor to your own interests and background based on what you're interested in and what you'd want to understand in more depth)

* "Why does feature scaling affect eigenvalues of $X^TX$ and thus GD step sizes?"
* "Show a toy case where normal equation is numerically unstable; compare to ridge."
* "Derive the bias introduced by L2 regularization and sketch the bias–variance tradeoff."
* "How would you detect under/overfitting with a validation curve in this setup?"
* "What breaks if features are perfectly collinear? How does ridge fix it?"
* "How do I pick a learning-rate schedule? Try constant vs. decay in a plot."


#### Reflection (<300 words, use this scaffold)

The best way to learn is to write down what you remember from readings, conversations and lectures. Here's a guide to do it yourself (submit this and don't worry about editing language too much; in fact, it's better if you don't edit it too much so that I know it's not a language model that generated it). Here are questions to consider in your reflection.

**What?** Briefly describe what you explored (topics, derivations, code).
**So what?** What clicked? What remained confusing? What errors did you make on MCQs and why?
**Now what?** Two concrete next steps to strengthen your understanding (e.g., try L1, regularization paths, condition numbers).

#### Grading rubric (20 pts total)

* **Conversation quality (5 pts):** Followed the flow, asked off-script questions, requested derivations; interactions are inquisitive and substantive.
* **MCQs (5 pts):** Completed **10** MCQs. How many you got correctly is important but only counts for 1 point (if you got most correctly). Exploration of deeper concepts and demonstration of understanding and curiosity is more important.
* **Code & plots (4 pts):** At least one runnable experiment with labeled plot(s); shows a relevant comparison (e.g., GD vs. SGD or λ sweep).
* **Reflection (4 pts):** Clear, honest, and actionable; ties mistakes to next steps.
* **Presentation (2 pts):** One clean PDF; readable code/plots; includes lecture link(s).

A few more points: 
* Late or incomplete items may lose points.
* Do your own conversation and reasoning. It’s fine if you discuss ideas with classmates, but **do not copy** chats or answers.
* If you used any external sources beyond the provided lecture (blogs, docs), list them at the end of your PDF.
* If an answer seems off, **ask the model to re-derive or verify** with a small numeric example. Be always skeptical!
* When you run code locally, compare your outputs to the model's and ask "why are these different?".
* If you ask for data-plotting examples, prefer tiny datasets (n<200) so plots render clearly and code runs fast

