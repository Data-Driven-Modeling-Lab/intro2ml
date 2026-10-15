---
title: "Track 3: can a recurrent network beat "tomorrow is like today"?"
layout: note
category: "Jupyter Notebook"
permalink: /materials/notebooks/lab_rnn_beirut/
notebook_source: "lab_rnn_beirut.ipynb"
colab_url: "https://colab.research.google.com/github/Data-Driven-Modeling-Lab/intro2ml/blob/main/materials/notebooks/lab_rnn_beirut.ipynb"
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Data-Driven-Modeling-Lab/intro2ml/blob/main/materials/notebooks/lab_rnn_beirut.ipynb)


**Introduction to Machine Learning, neural networks lab.** Upload this notebook to Colab (File, Upload notebook) or run it in Jupyter. CPU only, a few seconds per model.

**The question.** Beirut's daily maximum temperature, 2015 to 2025 (ERA5 reanalysis via Open-Meteo, CC-BY 4.0). Forecast tomorrow's maximum from the last 14 days. Train on 2015 to 2022; test on the 1,096 days of 2023 to 2025, which the model never sees.

**The score** is the root mean squared error on the test days, in degrees Celsius: lower is better. The honest baseline comes first: say tomorrow will be like today.


```python
import numpy as np, torch, torch.nn as nn, json, os, urllib.request

URL = 'https://learn.sematlas.com/static/data/datasets/beirut_weather.json'
if not os.path.exists('beirut_weather.json'):
    urllib.request.urlretrieve(URL, 'beirut_weather.json')
w = json.load(open('beirut_weather.json'))['daily']
days = np.array(w['time'])
tmax = np.array(w['temperature_2m_max'], dtype=float)
print(len(days), 'days from', days[0], 'to', days[-1], '; hottest', tmax.max(), 'C, coldest maximum', tmax.min(), 'C')

L = 14                                                    # days of history the model reads
target = tmax[L:]                                         # tomorrow's maximum, for every window
train_days = np.array([s < '2023-01-01' for s in days[L:]])
test_days = ~train_days
rmse = lambda pred, true: float(np.sqrt(np.mean((pred - true) ** 2)))

persistence = tmax[L - 1:-1]                              # the forecast "tomorrow = today"
print(f'persistence: {rmse(persistence[test_days], target[test_days]):.2f} C')
```

## 1. Choose what the model reads

`INPUTS` lists the daily series in each window. Start with the maximum alone; later add the night minimum, the rain, or the season (the day of the year as a point on a circle, so 31 December sits next to 1 January).


```python
INPUTS = ['tmax']            # try ['tmax', 'tmin'], or ['tmax', 'tmin', 'rain', 'season']

def series(name):
    if name == 'tmax':   return [tmax]
    if name == 'tmin':   return [np.array(w['temperature_2m_min'], dtype=float)]
    if name == 'rain':   return [np.log1p(np.array(w['precipitation_sum'], dtype=float))]
    if name == 'season':
        doy = np.array([np.datetime64(s, 'D').astype(object).timetuple().tm_yday for s in days])
        return [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]

F = np.stack([s for n in INPUTS for s in series(n)], 1)              # (days, features)
windows = np.stack([F[i:i + L] for i in range(len(F) - L)])          # (samples, 14, features)
print('windows', windows.shape)

# the linear model on the same windows: every day and feature gets one weight
A = np.c_[windows.reshape(len(windows), -1), np.ones(len(windows))]
coef = np.linalg.lstsq(A[train_days], target[train_days], rcond=None)[0]
print(f'linear on {L} days of {INPUTS}: {rmse(A[test_days] @ coef, target[test_days]):.2f} C')
```

## 2. A recurrent network

An LSTM reads the 14 days in order, one day at a time, with the same weights at every step, and keeps a hidden state of 16 numbers; a linear layer turns the last hidden state into the forecast. Inputs and target are standardized with the training days' mean and spread.


```python
mu, sd = F[L:][train_days].mean(0), F[L:][train_days].std(0)
t_mu, t_sd = target[train_days].mean(), target[train_days].std()
Xn = torch.tensor((windows - mu) / sd, dtype=torch.float32)
yn = torch.tensor((target - t_mu) / t_sd, dtype=torch.float32)

class Forecaster(nn.Module):
    def __init__(self, n_in, hidden=16, kind='LSTM'):
        super().__init__()
        self.rnn = getattr(nn, kind)(n_in, hidden, batch_first=True)   # kind: 'RNN', 'LSTM' or 'GRU'
        self.out = nn.Linear(hidden, 1)
    def forward(self, x):
        h, _ = self.rnn(x)                  # h: the hidden state after every day
        return self.out(h[:, -1]).squeeze(-1)

def fit(model, epochs=30, lr=0.01, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xa, ya = Xn[train_days], yn[train_days]
    for epoch in range(epochs):
        perm = torch.randperm(len(ya))
        for i in range(0, len(ya), 64):
            idx = perm[i:i + 64]
            loss = nn.functional.mse_loss(model(Xa[idx]), ya[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = model(Xn[test_days]).numpy() * t_sd + t_mu
    return pred

torch.manual_seed(0)
pred = fit(Forecaster(F.shape[1]))
print(f'recurrent network on {L} days of {INPUTS}: {rmse(pred, target[test_days]):.2f} C')
```


```python
# one month of the test period: the truth, persistence, and the network
import matplotlib.pyplot as plt
td = days[L:][test_days]
s = np.flatnonzero(td == '2024-03-01')[0]; e = s + 31
plt.figure(figsize=(9, 3))
plt.plot(target[test_days][s:e], 'k.-', label='measured')
plt.plot(persistence[test_days][s:e], ':', label='tomorrow = today')
plt.plot(pred[s:e], label='LSTM')
plt.ylabel('daily maximum (C)'); plt.xlabel('day of March 2024'); plt.legend(); plt.show()
```

## 3. Your turn

On our runs: persistence 2.21 C, the linear model on 14 days of maxima 2.05 C, this LSTM about 1.96 C. Can you get lower, and does the network still beat the linear model when both read the same inputs?

1. Add inputs: `INPUTS = ['tmax', 'tmin', 'rain', 'season']`, and rerun both cells.
2. `kind='GRU'` or `kind='RNN'`; `hidden=32`; `L = 7` or `L = 30`.
3. Run each with two seeds before you believe a difference of 0.02 C.
4. Look at the March plot: when does every model miss? What is not in the file that the weather knew?

**Report** your best test RMSE, what the model read, and the linear model's number on the same inputs.
