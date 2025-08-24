---
title: "Generating data"
layout: note
category: "Jupyter Notebook"
permalink: /materials/notebooks/basic_fit_example/
notebook_source: "basic_fit_example.ipynb"
---

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
%matplotlib widget

np.random.seed(0)
x = np.linspace(-3, 3, 30)
y = 3.5 * x + np.random.normal(size=x.size)

# Initial hypothesis parameters
m_init = 2
b_init = 0

# Function to calculate hypothesis y = mx + b
def hypothesis(x, m, b):
    return m * x + b

# Create the initial plot with adjusted axes for more space at the bottom
fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'bottom': 0.25})

# Scatter and hypothesis line
ax[0].scatter(x, y, color='blue', label='Data')
line, = ax[0].plot(x, hypothesis(x, m_init, b_init), color='red', label='Hypothesis')
# Initial dotted lines
dotted_lines = [ax[0].plot([xi, xi], [yi, hypothesis(xi, m_init, b_init)], 'k:', linewidth=1) for xi, yi in zip(x, y)]
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].legend()

# Contour plot of loss
m_range = np.linspace(-1, 5, 400)
b_range = np.linspace(-10, 10, 400)
M, B = np.meshgrid(m_range, b_range)
Z = np.array([((y - hypothesis(x, mi, bi))**2).mean() for mi, bi in zip(np.ravel(M), np.ravel(B))])
Z = Z.reshape(M.shape)
contour = ax[1].contourf(M, B, Z, levels=50, cmap='rainbow')
point, = ax[1].plot([m_init], [b_init], 'ro')  # Point on contour plot
ax[1].set_xlabel(r'$\theta_0$')
ax[1].set_ylabel(r'$\theta_1$')
fig.colorbar(contour, ax=ax[1], label='Loss')

# Adjusting slider positions
axcolor = 'lightgoldenrodyellow'
ax_m = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)  # More space from the bottom
ax_b = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)

s_m = Slider(ax_m, r'$\theta_0$', -1, 5, valinit=m_init)
s_b = Slider(ax_b, r'$\theta_1$', -10, 10, valinit=b_init)

# Update function
def update(val):
    m = s_m.val
    b = s_b.val
    line.set_ydata(hypothesis(x, m, b))
    point.set_data([m], [b])  # Use list for coordinates
    # Update dotted lines
    for dotted_line, xi, yi in zip(dotted_lines, x, y):
        dotted_line[0].set_ydata([yi, hypothesis(xi, m, b)])
    fig.canvas.draw_idle()

s_m.on_changed(update)
s_b.on_changed(update)

# Function to handle mouse clicks on the contour plot
def onclick(event):
    if event.inaxes == ax[1]:
        m, b = event.xdata, event.ydata
        s_m.set_val(m)
        s_b.set_val(b)

# Connect the click event
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# plt.tight_layout()
plt.show()
```


    ![png](/materials/notebooks/basic_fit_example/output_1_0.png)

```python

```


```python

```
