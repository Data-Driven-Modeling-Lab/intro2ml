---
title: "Constants"
layout: note
category: "Reference Material"
permalink: /materials/notebooks/value_iteration/
notebook_source: "value_iteration.ipynb"
---

```python
import numpy as np

# Constants
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
actions = [UP, DOWN, LEFT, RIGHT]

# Grid dimensions
rows, cols = 4, 4
states = [(i, j) for i in range(rows) for j in range(cols) if not (i == 1 and j == 1)]

# Rewards and transitions
rewards = {state: 0 for state in states}  # Initialize rewards
rewards[(0, 2)] = 1    # Treat
rewards[(2, 1)] = 3    # Treat
rewards[(1, 3)] = 10   # Ball
rewards[(2, 2)] = -5   # Cat
rewards[(2, 0)] = -5   # Cat
rewards[(3, 3)] = 20   # Child

# Value function
V = {state: 0 for state in states}

# Discount factor
gamma = 0.9

# Transition probabilities
prob_success = 0.8
prob_fail = 0.2

def action_result(state, action):
    """ Given state and action, return the resulting state based on the action's direction """
    i, j = state
    if action == UP:
        return (max(i-1, 0), j)
    elif action == DOWN:
        return (min(i+1, rows-1), j)
    elif action == LEFT:
        return (i, max(j-1, 0))
    elif action == RIGHT:
        return (i, min(j+1, cols-1))
    return state

def value_iteration(V, rewards, theta=0.0001):
    """ Value iteration algorithm """
    while True:
        delta = 0
        for state in states:
            if state in [(3, 3)]:  # Terminal states
                continue
            v = V[state]
            max_value = float('-inf')
            for action in actions:
                total = 0
                # Main action result

                main_result = action_result(state, action)
                # Calculate value for main action

                total += prob_success * (rewards.get(main_result, 0) + gamma * V.get(main_result, 0))
                # Calculate value for staying in the same place (failure case)
                total += prob_fail * (rewards.get(state, 0) + gamma * V.get(state, 0))
                max_value = max(max_value, total)
            V[state] = max_value
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V

# Perform value iteration
V = value_iteration(V, rewards)

# Display the values
for i in range(rows):
    for j in range(cols):
        if (i, j) in V:
            print(f"Value of ({i}, {j}): {V[(i, j)]:.2f}")
        else:
            print("Wall")
    print()
```

    Value of (0, 0): 961.87
    Value of (0, 1): 974.02
    Value of (0, 2): 985.31
    Value of (0, 3): 997.50
    
    Value of (1, 0): 949.88
    Wall
    Value of (1, 2): 997.50
    Value of (1, 3): 999.99
    
    Value of (2, 0): 956.99
    Value of (2, 1): 967.31
    Value of (2, 2): 983.81
    Value of (2, 3): 997.50
    
    Value of (3, 0): 946.29
    Value of (3, 1): 958.24
    Value of (3, 2): 966.56
    Value of (3, 3): 0.00
    


```python
V
```


    {(0, 0): 961.8747637603137,
     (0, 1): 974.0197715476114,
     (0, 2): 985.3080243827286,
     (0, 3): 997.4963814203093,
     (1, 0): 949.8812885152631,
     (1, 2): 997.4963814203093,
     (1, 3): 999.9901470063442,
     (2, 0): 956.989709794409,
     (2, 1): 967.3053595956482,
     (2, 2): 983.8118635610443,
     (2, 3): 997.4964799502459,
     (3, 0): 946.2884813159646,
     (3, 1): 958.2366901320638,
     (3, 2): 966.5573274644751,
     (3, 3): 0}


