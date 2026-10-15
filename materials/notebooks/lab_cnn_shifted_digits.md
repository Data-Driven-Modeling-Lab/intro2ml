---
title: "Track 2: a convolutional network for the shifted digits"
layout: note
category: "Jupyter Notebook"
permalink: /materials/notebooks/lab_cnn_shifted_digits/
notebook_source: "lab_cnn_shifted_digits.ipynb"
colab_url: "https://colab.research.google.com/github/Data-Driven-Modeling-Lab/intro2ml/blob/main/materials/notebooks/lab_cnn_shifted_digits.ipynb"
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Data-Driven-Modeling-Lab/intro2ml/blob/main/materials/notebooks/lab_cnn_shifted_digits.ipynb)


**Introduction to Machine Learning, neural networks lab.** Upload this notebook to Colab (File, Upload notebook) or run it in Jupyter. It runs on a CPU in a few seconds per model; no GPU needed.

**The challenge.** The network from the neural networks lecture reads about 91% of held-out digits, and about 25% once each digit moves two pixels. Build one that reads both.

**The rules.** 3,000 MNIST digits at 16 by 16 pixels. Every fifth digit (600) is held out; you train on the other 2,400 and never on the held-out ones. The held-out digits are scored twice: as they are, and each moved two pixels up, down, left or right (the direction is fixed in advance). Your **score is the lower of the two accuracies**. The data, split, shift and score are the same as on the slides, so your numbers go on the same board.


```python
import numpy as np, pandas as pd, torch, torch.nn as nn, os, urllib.request

URL = 'https://learn.sematlas.com/static/data/datasets/mnist_16x16.csv'
if not os.path.exists('mnist_16x16.csv'):
    urllib.request.urlretrieve(URL, 'mnist_16x16.csv')
d = pd.read_csv('mnist_16x16.csv')
X, y = d.drop(columns='label').values / 255, d.label.values
test = np.arange(len(y)) % 5 == 0                     # every fifth digit is held out

def shift(X, dx, dy):
    """Move 16 by 16 images dx pixels right and dy pixels down; ink that leaves the frame is lost."""
    img = X.reshape(-1, 16, 16); out = np.zeros_like(img)
    out[:, max(dy,0):16+min(dy,0), max(dx,0):16+min(dx,0)] = img[:, max(-dy,0):16+min(-dy,0), max(-dx,0):16+min(-dx,0)]
    return out.reshape(len(X), 256)

X_test, y_test = X[test], y[test]
pick = np.random.default_rng(0).integers(0, 4, len(y_test))   # one direction per held-out digit, fixed
X_shift = X_test.copy()
for k, (dx, dy) in enumerate([(2, 0), (-2, 0), (0, 2), (0, -2)]):
    X_shift[pick == k] = shift(X_test[pick == k], dx, dy)

as_images = lambda A: torch.tensor(A, dtype=torch.float32).reshape(-1, 1, 16, 16)
Xtr, ytr = as_images(X[~test]), torch.tensor(y[~test])
Xte, Xsh, yte = as_images(X_test), as_images(X_shift), torch.tensor(y_test)
print('train', tuple(Xtr.shape), ' held out', tuple(Xte.shape), ' shifted', tuple(Xsh.shape))

def score(model, note=''):
    model.eval()
    with torch.no_grad():
        clean = (model(Xte).argmax(1) == yte).float().mean().item()
        moved = (model(Xsh).argmax(1) == yte).float().mean().item()
    n = sum(p.numel() for p in model.parameters())
    print(f'clean {clean:.3f}   shifted {moved:.3f}   score {min(clean, moved):.3f}   ({n:,} weights) {note}')
    return min(clean, moved)
```

## 1. The baseline: the lecture's network in PyTorch

256 pixels, 32 hidden ReLU units, 10 outputs. `train` is the loop from the lecture: forward, `zero_grad`, `backward`, `step`. With `augment=True` every batch is moved by a random shift of up to two pixels before the step.


```python
def shift_batch(xb, dx, dy):
    out = torch.zeros_like(xb)
    out[..., max(dy,0):16+min(dy,0), max(dx,0):16+min(dx,0)] = xb[..., max(-dy,0):16+min(-dy,0), max(-dx,0):16+min(-dx,0)]
    return out

def train(model, epochs=30, lr=0.005, augment=False, seed=0):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()                 # softmax and cross-entropy in one
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr), generator=g)
        for i in range(0, len(ytr), 32):
            idx = perm[i:i + 32]
            xb, yb = Xtr[idx], ytr[idx]
            if augment:                              # a random move of up to two pixels, for the whole batch
                dx, dy = torch.randint(-2, 3, (2,), generator=g).tolist()
                xb = shift_batch(xb, dx, dy)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

torch.manual_seed(0)
mlp = nn.Sequential(nn.Flatten(), nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, 10))
score(train(mlp), 'the lecture network')
```

You should see about 0.91 on the digits as they are and about 0.25 once they move: a fully connected layer gives every pixel its own weight, so a stroke learned in one place means nothing two pixels over.

## 2. A convolutional network

Two convolutions with 3 by 3 kernels, each followed by ReLU and a 2 by 2 max pool, then one linear layer. `padding=1` keeps the 16 by 16 size through each convolution; each pool halves it: 16, 8, 4.


```python
def make_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 8 maps of 8 by 8
        nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16 maps of 4 by 4
        nn.Flatten(), nn.Linear(16 * 4 * 4, 10))

cnn = train(make_cnn())
score(cnn, 'CNN')
```

Fewer weights than the baseline, and better on both tests. But look at the shifted number: about half. The same kernel slides over the whole image, so a stroke is detected wherever it is; but the last layer still gives each of the 4 by 4 positions its own weights, so the network as a whole has not learned that a 2 moved up is still a 2.

## 3. Your turn

Start with the one change that tells the network what you want it to ignore: `augment=True`. Then try one change at a time and write down both numbers each time.


```python
cnn_aug = train(make_cnn(), augment=True)
score(cnn_aug, 'CNN, trained on shifted batches')
```

Ideas, roughly in order of effort:

1. More detectors: `Conv2d(1, 16, ...)` and `Conv2d(16, 32, ...)` (change `16 * 4 * 4` to match).
2. More epochs, or a smaller learning rate for the last ten.
3. A third convolution, or global average pooling (`nn.AdaptiveAvgPool2d(1)`) instead of the last pool and flatten: what does it do to the shifted score?
4. Shift each digit by its own random amount instead of the whole batch by one.
5. Does augmentation alone rescue the fully connected network? Train `mlp` with `augment=True`.

Before you report a number, run it with a second `seed`: a difference smaller than the gap between two seeds is not a difference.


```python
# Which shifted digits does your model still get wrong? (needs matplotlib, which Colab has)
import matplotlib.pyplot as plt
model = cnn_aug
with torch.no_grad():
    pred = model(Xsh).argmax(1)
wrong = torch.nonzero(pred != yte).flatten()[:12]
fig, axes = plt.subplots(1, len(wrong), figsize=(1.3 * len(wrong), 1.7))
for ax, i in zip(np.atleast_1d(axes), wrong):
    ax.imshow(Xsh[i, 0], cmap='gray'); ax.axis('off')
    ax.set_title(f'{yte[i].item()} as {pred[i].item()}', fontsize=9)
plt.show()
print(len(torch.nonzero(pred != yte)), 'of', len(yte), 'shifted digits misread')
```

**Report** your best run on the class board: the two numbers, the score, the number of weights, and one sentence on what you changed. Keep the run that gave it, so you can show it.
