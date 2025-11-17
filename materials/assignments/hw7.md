---
title: "Problem Set 7: Reinforcement Learning"
layout: note
category: "Assignment"
permalink: /materials/assignments/hw7/
---


**Instructions:** Submit the solution as a PDFs and notebooks on the following [submission link](https://lms.aub.edu.lb/mod/assign/view.php?id=2428505)

## Problem 1: Proving the Convergence of Value Iteration (50 points)

Consider a Markov Decision Process (MDP) with finite state and action spaces, and a discount factor $0 < \gamma < 1$. In class we discussed that the value iteration algorithm converges to the optimal value function. In this problem, we will prove this result.

Let $B$ denote the Bellman update operator, which maps a value function $V$ (a vector assigning a real value to each state) to a new value function $B(V) \equiv V'$ defined as:

$$
V'(s) = R(s) + \gamma \max_{a \in A} \sum_{s' \in S} P(s'|s,a) V(s'),
$$

where:

* $R(s)$ is the reward for being in state $s$,
* $P(s' \mid s,a)$ is the transition probability from state $s$ to $s'$ after taking action $a$.

a) Show that for any two finite-valued vectors $V_1$ and $V_2$, the following inequality holds:

$$
\|B(V_1) - B(V_2)\|_\infty \le \gamma \|V_1 - V_2\|_\infty,
$$

where $\| V \|_{\infty} = \max_{s \in S} \| V(s) \|$.

b) A vector $V$ is called a fixed point of $B$ if it satisfies $B(V) = V$. Using the result above, prove that $B$ has at most one fixed point; that is, there can be at most one solution to the Bellman equations. You can assume that at least one fixed point exists.

c) Starting from [this example code](https://colab.research.google.com/github/Data-Driven-Modeling-Lab/intro2ml/blob/main/materials/references/value_iteration_new.ipynb), show that V decreases monotonically with each iteration of value iteration.

## Problem 2: MDPs for Real-World Control and Decision-Making (50 points)

In this exercise, you will practice formulating **Markov Decision Processes (MDPs)** for real-world control and decision-making scenarios. Each MDP should be described by the tuple
$$
\mathcal{M} = (S, A, P, R, \gamma),
$$

where:

* $S$ is the **state space**
* $A$ is the **action space**
* $P(s' \mid s,a)$ is the **transition probability**
* $R(s,a)$ is the **reward function**
* $\gamma$ is the **discount factor**

1. Identify what the **state**, **action**, and **reward** should represent.
2. Write the sets $ S, A, R $ explicitly

Note that the solutions are not unique. Feel free to consider multiple state, action, and reward functions as long as they are consistent with the problem statement, and briefly explain your choices. 

Here's the pendulum problem you saw in class, with a slightly different state and action space.

#### Example: Inverted Pendulum on a Cart

A classic control problem involves balancing a pendulum on a moving cart by applying horizontal forces. We keep only **cart position** and **pole angle** as the state. Actions are **discrete force commands** on the cart.

1. **State**: Cart position: $ X=\{x_1,\dots,x_{N_x}\} $, with $ N_x $ being the number of discrete positions. 

2. **Action space (discretized force)**: Choose a small, finite set of forces:
 $$ A=\{-F_{\max},-\tfrac{F_{\max}}{2},0,\tfrac{F_{\max}}{2},F_{\max}\} $$.

3. **Reward function**: Penalize angle deviation: $R(s,a) = -\alpha\theta^{2}, \qquad \alpha>0.$

---

*Note:* that typically in the pendulum, the transition probability is replaced by a physics model that takes the form $ s_{t+1} = f(s_t, a_t) + \varepsilon_t $, where $ f $ is a function that takes the state and action and returns the next state, and $ \varepsilon_t $ is a noise term. In this problem, you do not need to specify the detailed transition probabilities. You may assume there exists a probability distribution $ P(s' \mid s,a) $ induced by a one-step physics update. The transition probability is given by:

$$
P(s' \mid s,a) = \begin{cases}
1 & \text{if } s' = f(s,a) + \varepsilon_t \\
0 & \text{otherwise}
\end{cases}
$$

This is the first step for generalizing MDPs to continuous state and action spaces.

---

**(a) Autonomous Drone Navigation**: A drone must fly from a starting point to a target location while avoiding obstacles and minimizing energy consumption.

**(b) Self-Driving Car at an Intersection**: A self-driving car must decide whether to stop, go, or slow down based on traffic light color and nearby vehicles. *Hint:* State includes distance to intersection, light state, and car velocity.
Actions may be discrete choices. Reward could penalize collisions or red-light crossings.

**(c) Warehouse Robot Picking Items**: A robot moves in a grid warehouse to collect items and deliver them to packing stations efficiently.

**(d) Smart Thermostat Control**: A thermostat adjusts the temperature of a room based on external temperature, time of day, and occupancy patterns.

**(e) Investment Portfolio Management**: An agent allocates wealth across several assets to maximize long-term returns.
