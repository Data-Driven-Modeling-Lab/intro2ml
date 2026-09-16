---
title: "Problem Set 1: Linear Regression"
layout: note
category: "Assignment"
permalink: /materials/assignments/hw1/
---


This is an individual assignment. I encourage you to discuss the problems with each other but the final write up has to be yours. Copying or sharing code is not allowed. The work you hand in has to be yours, and you have to
be able to explain every line of it. Review the [assignment policy](https://intro2ml.com/logistics/) on collaboration and late
submissions before you start.

Submission guidelines: Problem 0 should be done on Slack, Problem 1 and 5 on the slides, problem 2, 3 and 4 zipped and submitted to moodle (link below), problem 4 should also be submitted on Kaggle (more details in the problem description).

Deadlines: The homework is due on **Wednesday 23 September at 23:59**. Zip what goes to Moodle into a single file and submit it here:
[Submission Link](https://lms.aub.edu.lb/mod/assign/view.php?id=2827253&forceview=1)

## Problem 0: Some AI News and an invitation to a forum discussion (10 points)

On 8 September 2026, OpenAI announced that a swarm of 10,000 agents,
using a model it hasn't released yet, came up with a proof about the
three-dimensional Navier-Stokes equations in 88 hours, followed by
17 hours of formalization in Lean (an non-ML algorithmic proof system used by mathematicians to formalize or derive theorems). Much of the press called it a solution to one of the seven Millennium Prize Problems; each of which, when solved, awards the winner 
one million dollars! OpenAI says it will not claim the prize.

Before you tell me what you think, four details that the headlines mostly skipped.

To start with, what is the Navier-Stokes equation(s)? The equations that describe how fluids move: water
from a tap, blood in an artery, air over a wing, the weather. A *singularity*,
or blow up, means a solution stops being smooth after a finite time: something
like the velocity gradient goes to infinity. If that can happen, the
equations stop describing the fluid at that instant. The Millennium Problem asks
for a proof of any ONE of four statements that Fefferman wrote down for the Clay
Institute: that smooth solutions always exist, either in all of space (A) or in a
periodic box (B), with no external force; or that they can break down, again in
space (C) or in a box (D), where you are allowed to apply a smooth force. Any one
of the four wins the prize.

What the paper claims to prove, and it has not been reviewed yet, is (C) and (D):
a fluid starting from rest, with a smooth force applied and finite energy the whole
way, that develops a singularity in finite time. That is one of the four
statements, and according to the problem statement it counts.

Many people consider (A) and (B), the unforced versions, more interesting. But they
are not required to consider the problem, as stated, solved.

A little more about Lean. Lean is a proof assistant: you state your definitions
and your claim formally, and the software mechanically verifies every step. A
Lean-checked proof is a different kind of evidence from a human-written, typically prose, 
argument. Lean ensures that all arguments are rigorous, which partially solves the ambiguity and halluciation problem with language models. It's not the same as peer review, which for a proof this long will probably take months to do; typically, reviews take months even if the paper is 10 pages long! 

There's a bit of drama behidn this news. Tristan Buckmaster (NYU) and Levent Alpoge (Anthropic) say
they spent around a year on related fluid blow-up problems and had
Lean-verified results on the related (Boussinesq and three-dimensional Euler) equations
before 22 August. Buckmaster has suggested OpenAI has taken on his private
work stored in Codex (OpenAI's coding agent). OpenAI's published page credits both for concurrent work
on the forced Euler problem and has offered a joint announcement.

Some places to read, and you should find your own as well:
[OpenAI's own writeup](https://openai.com/index/navier-stokes-solution/),
[Nature](https://www.nature.com/articles/d41586-026-02842-5),

### What to discuss

Start a conversation on the `#discussion-forum`. This is open ended and there is no correct answer. I want to see real engagement, the formation of an opinion, sharing relevant resources that might help you and others understand what's happening, etc. Feel free to focus on whatever part of the story you're interested in (technical, ethical, technological, mathematical, etc), and try to express a position clearly and succinctly. In this case, **I discourage using LLMs** to completely write you posts for you. I want to hear your way of phrasing things, and putting words together. I much prefere a brief answer that sounds like you, than a long paragraph that sounds like a machine. In this specific context where you're expecting to have the conversation with other humans, authenticity matters.

Some tips about where to take the conversation: what's the context for the finding? Does it generalize to other theorems? What are the technical details of the proof? etc.

## Problem 1: MCQ warm up (20 points)

Solve the following MCQ on the course slides. Work through the PS1 warm up
deck and answer the questions in place. Your answers are recorded as you go, so
there is nothing to submit and nothing to zip. Here's the link [PS1 warm up](https://learn.sematlas.com/slides/ps1#/warmup)

I recommend going through them after reading the lecture notes to test your understanding.

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

Submit it on Moodle as `problem_2.ipynb`, with the dataset in the same folder so that it runs locally. If it does not run, you will lose points.

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

## Problem 4: Kaggle competition (20 points)

**Enter on your own.** No teams this year: the leaderboard should tell each of
you where your own model stands. Talking through approaches with classmates is
fine and encouraged; handing round code or submission files is not.

Sign up at kaggle.com with a username that is your first name and last name. Join the competition
through this link: [Competition Link](https://www.kaggle.com/competitions/find-the-polynomial-ml26-27). Please do not share
the link outside the class for now. The problem is to fit a function given two
real inputs and one real output. Use whatever method you see fit. You are
allowed two submissions per day, and you will find more detail on the
competition page. Along with the kaggle submissions, submit your code containing the solutions you've tried on Moodle as `problem_4.ipynb`.

## Problem 5: Learning machine learning with machine learning (20 points)

The best way to learn is to try and recall everything you've learned without using any reference. You'll try to rederive things if you've forgotten them, and you'll develop your own perspective on seeing things. This works even if you're not taking a course; if you want to learn about any topic on your own. In this exercise, we're leveraging the fact that you can try to recall details through a conversation about linear regression with a 
course tutor. Explaining the core ideas in your own
words, asking for derivations, getting it to write and run a small experiment,
testing yourself with its questions, and reflecting on what you got wrong.

Link: [Learning ML with ML](https://learn.sematlas.com/slides/ps1#/chat-lmlml)

You open, not the tutor. Start with a reflection written from memory, around
300 words: what you explored, what clicked and what did not, and two examples
you would like to think about more. Write mathematical expressions when the
words are vague. The tutor reads it, tells you what was solid and what was
vague, and then gives you five questions one at a time, built from what you
wrote, holding back each explanation until you have answered. After that,
steer it wherever you want, as long as it is machine learning: ask for the
technical details when its words are vague, ask for code and a plot, take it
off script.

The conversation is recorded as you have it, so there is no transcript to
export and no PDF to assemble.

Reminder: Be skeptical of what a model tells you. If an answer seems off, ask it to
  re-derive the result, check it in the notes, or compare
  its output to what you get when you run the code yourself.

