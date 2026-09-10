---
title: "Problem Set 1: Linear Regression"
layout: note
category: "Assignment"
permalink: /materials/assignments/hw1/
published: false   # under revision for F26
---


This is an individual assignment, apart from the Kaggle competition. Review the
[assignment policy](https://intro2ml.com/logistics/) on collaboration and late
submissions before you start.

**Where each problem is done.** Two of the six problems happen on the course
slides site rather than on paper, and one happens on Slack:

| Problem | Where | What you hand in |
|---|---|---|
| 0. Forum discussion | Slack, `#learn-and-share` | One substantive post and one reply |
| 1. MCQ warm up | Course slides | Answered in place, recorded automatically |
| 2. Weather dataset | Moodle | `problem_2.ipynb` |
| 3. Regression and gradient descent | Moodle | `problem_3.pdf` |
| 4. Kaggle competition | Kaggle | Your team's submissions |
| 5. Learning ML with ML | Course slides | The conversation itself, recorded automatically |

Zip what goes to Moodle into a single file and submit it here:
[Submission Link](TODO_MOODLE_LINK)

## Problem 0: Forum discussion, did an AI just solve a Millennium Prize problem? (10 points)

On 8 September 2026, OpenAI announced that a swarm of roughly 10,000 agents,
running a model it has not released, produced a proof about the
three-dimensional Navier-Stokes equations in about 88 hours, followed by
roughly 17 hours of formalization in Lean. Much of the press called it a
solution to one of the seven Millennium Prize Problems, each of which carries a
one million dollar prize. OpenAI says it will not claim the prize.

Before you argue about it, four details that the headlines mostly skipped.

**What Navier-Stokes is.** The equations that describe how fluids move: water
from a tap, blood in an artery, air over a wing, the weather. A *singularity*,
or blow up, means a solution stops being smooth after a finite time: something
like the velocity gradient runs off to infinity. If that can happen, the
equations stop describing the fluid at that instant. The Millennium Problem
asks whether smooth solutions always stay smooth.

**What was actually proved.** Not the Millennium Problem as stated. The result
shows a singularity can form in finite time in the three-dimensional equations
*with a smooth forcing term added*. The Millennium Problem concerns the
unforced equations. Whether the forced result is a real step toward the
unforced one, or a different question wearing its clothes, is a live argument
among people who work on this.

**How it was checked.** Lean is a proof assistant: you state your definitions
and your claim formally, and the software mechanically verifies every step. A
Lean-checked proof is a different kind of evidence from a long, fluent,
convincing-sounding argument in prose, which is exactly the failure mode people
worry about with language models. It is also not the same thing as peer review,
which for a claim this size usually takes months. The proof was released on 8
September, so that review has barely started.

**Who found it.** Tristan Buckmaster (NYU) and Levent Alpoge (Anthropic) say
they had spent about a year on related fluid blow-up problems and had
Lean-verified results on the Boussinesq and three-dimensional Euler equations
by 22 August. Buckmaster has suggested OpenAI may have drawn on his private
work stored in Codex. OpenAI's published page credits both for concurrent work
on the forced Euler problem and has offered a joint announcement.

Some places to read, and you should find your own as well:
[OpenAI's own writeup](https://openai.com/index/navier-stokes-solution/),
[Nature](https://www.nature.com/articles/d41586-026-02842-5),
[Washington Post](https://www.washingtonpost.com/technology/2026/09/09/openai-claims-it-solved-elusive-math-problem-with-1-million-prize/),
[Semafor](https://www.semafor.com/article/09/08/2026/openai-agents-find-proof-to-1-million-millennium-prize-problem).

### What to discuss

Post on `#learn-and-share`. This is open ended and there is no correct answer;
I am marking the quality of the thinking, not the position you land on. Pick
whatever grabs you, and take a position rather than surveying both sides.

Some things worth arguing about:

- **Is this a finding?** What would have to be true for you to call it one.
  Does it matter that the theorem proved is not the theorem that carries the
  prize? Does it matter that no human can hold the whole argument in their
  head?
- **What counts as verification.** In this course you fit a model and then
  check it on data the model never saw. A proof has no test set. Lean checks
  that the steps follow from the assumptions, but not that the assumptions are
  the interesting ones. Which of these is the stronger guarantee, and of what?
- **Where the finding came from.** If a model is shown someone's unpublished
  work and then produces a related result, who found it? Compare this to how
  you will use an LLM in Problem 5 of this very problem set. When does using a
  tool become the tool doing it?
- **Prospect and risk.** If this generalizes, what happens to a field like
  fluid dynamics, or to the training of the next generation of researchers?
  What is lost, if anything? And what would a *bad* version of this look like:
  what kind of claim would be much harder to catch than a wrong proof?

### What to hand in

Nothing on Moodle. Marked on Slack participation:

- **One substantive post** (roughly 150 to 400 words) taking a position and
  giving a reason for it. Cite something you actually read.
- **At least one reply** to a classmate that moves their argument forward:
  extend it, complicate it, or disagree with a reason. "Good point, I agree" is
  not a reply.

Posts that quote a chatbot's summary of the story without adding a thought of
your own will not get credit. Read something.

## Problem 1: MCQ warm up (20 points)

This one is on the course slides, not on paper. Work through the PS1 warm up
deck and answer the questions in place. Your answers are recorded as you go, so
there is nothing to submit and nothing to zip.

Link: TODO_SLIDES_MCQ_LINK

The questions cover least squares, feature scaling, collinearity, polynomial
degree and test error, ridge, cross-validation, early stopping, and gradient
descent convergence. Answer them after you have done the reading and the
lecture, not before: the point is to find out what you actually absorbed.

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

Sign up at kaggle.com and form teams of up to three. Join the competition
through this link: [Competition Link](TODO_KAGGLE_LINK). Please do not share
the link outside the class for now. The problem is to fit a function given two
real inputs and one real output. Use whatever method you see fit. You are
allowed two submissions per day, and you will find more detail on the
competition page.

## Problem 5: Learning machine learning with machine learning (20 points)

Also on the course slides. You will run one sustained conversation with the
course tutor about linear regression: explaining the core ideas in your own
words, asking for derivations, getting it to write and run a small experiment,
testing yourself with its questions, and reflecting on what you got wrong.

Link: TODO_SLIDES_CONVERSATION_LINK

The conversation is recorded as you have it, so there is no transcript to
export and no PDF to assemble. What is marked is the quality of the
conversation: whether you pushed past the first answer, asked for the maths
when the words were vague, argued back when something sounded wrong, and were
honest in the reflection at the end. Length is not the point. A conversation
where you disagreed with the model twice is worth more than one three times as
long where you accepted everything.

## Notes

- Late or incomplete items may lose points.
- Do your own work and your own reasoning. Discussing ideas with classmates is
  fine; copying chats, notebooks or answers is not.
- List any external sources you used beyond the lectures.
- Be skeptical of what a model tells you. If an answer seems off, ask it to
  re-derive the result or check it against a small numeric example, and compare
  its output to what you get when you run the code yourself.
