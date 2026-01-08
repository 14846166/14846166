# -*- coding: utf-8 -*-
"""
HW7 - Gradient Descent for Sine Wave Fitting
Analytic gradient vs Numeric gradient
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# Utility function
# --------------------------------------------------
def scatter_pts_2d(x, y):
    """Set plotting limits with margin"""
    xmax = np.max(x)
    xmin = np.min(x)
    xgap = (xmax - xmin) * 0.2
    xmin -= xgap
    xmax += xgap

    ymax = np.max(y)
    ymin = np.min(y)
    ygap = (ymax - ymin) * 0.2
    ymin -= ygap
    ymax += ygap

    return xmin, xmax, ymin, ymax


# --------------------------------------------------
# Load dataset
# --------------------------------------------------
dataset = pd.read_csv('data/hw7.csv').to_numpy(dtype=np.float64)
x = dataset[:, 0]
y = dataset[:, 1]


# --------------------------------------------------
# Hyperparameters
# --------------------------------------------------
alpha = 0.05
max_iters = 500

# initial weights
w_init = np.array([-0.1607108, 2.0808538, 0.3277537, -1.5511576])


# --------------------------------------------------
# Analytic Gradient Descent
# --------------------------------------------------
w = w_init.copy()

for _ in range(1, max_iters):

    # model prediction
    y_hat = w[0] + w[1] * np.sin(w[2] * x + w[3])
    r = y - y_hat

    # analytic gradients
    grad = np.zeros_like(w)
    grad[0] = -2 * np.sum(r)
    grad[1] = -2 * np.sum(r * np.sin(w[2] * x + w[3]))
    grad[2] = -2 * np.sum(r * w[1] * np.cos(w[2] * x + w[3]) * x)
    grad[3] = -2 * np.sum(r * w[1] * np.cos(w[2] * x + w[3]))

    # update
    w = w - alpha * grad


xmin, xmax, ymin, ymax = scatter_pts_2d(x, y)
xt = np.linspace(xmin, xmax, 100)
yt1 = w[0] + w[1] * np.sin(w[2] * xt + w[3])


# --------------------------------------------------
# Numeric Gradient Descent (Finite Difference)
# --------------------------------------------------
w = w_init.copy()
eps = 1e-6

for _ in range(1, max_iters):

    grad = np.zeros_like(w)

    for k in range(len(w)):
        w1 = w.copy()
        w2 = w.copy()
        w1[k] += eps
        w2[k] -= eps

        y1 = w1[0] + w1[1] * np.sin(w1[2] * x + w1[3])
        y2 = w2[0] + w2[1] * np.sin(w2[2] * x + w2[3])

        J1 = np.sum((y - y1) ** 2)
        J2 = np.sum((y - y2) ** 2)

        grad[k] = (J1 - J2) / (2 * eps)

    # update
    w = w - alpha * grad


yt2 = w[0] + w[1] * np.sin(w[2] * xt + w[3])


# --------------------------------------------------
# Plot results
# --------------------------------------------------
plt.figure(dpi=288)
plt.scatter(x, y, color='k', edgecolor='w', linewidth=0.9, s=60, zorder=3)
plt.plot(xt, yt1, linewidth=4, c='b', label='Analytic method')
plt.plot(xt, yt2, linewidth=2, c='r', label='Numeric method')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()
