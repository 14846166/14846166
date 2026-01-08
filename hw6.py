# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import math
import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt
import pandas as pd


# calculate the eigenvalues and eigenvectors of a squared matrix
# the eigenvalues are decreasing ordered
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    # lambdas, V may contain complex value
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1] 
    return lambdas[sorted_idx], V[:, sorted_idx]


# class 1
mean1 = np.array([0, 5])
sigma1 = np.array([[0.3, 0.2],[0.2, 1]])
N1 = 200
X1 = np.random.multivariate_normal(mean1, sigma1, N1)

# class 2
mean2 = np.array([3, 4])
sigma2 = np.array([[0.3, 0.2],[0.2, 1]])
N2 = 100
X2 = np.random.multivariate_normal(mean2, sigma2, N2)

# m1: mean of class 1
# m2: mean of class 2
m1 = np.mean(X1, axis = 0, keepdims=1)
m2 = np.mean(X2, axis = 0, keepdims=1)

# write you code here
# LDA: 計算最佳投影方向
# Within-class scatter matrices
S1 = (X1 - m1).T @ (X1 - m1)
S2 = (X2 - m2).T @ (X2 - m2)
Sw = S1 + S2

# 投影方向：w = Sw^(-1) @ (m1 - m2)^T
w = la.inv(Sw) @ (m1 - m2).T
w = w.flatten()

# 正規化方向向量
w = w / la.norm(w)

# 計算分隔線（垂直於投影方向）
# 分隔線通過兩類別均值的中點
midpoint = (m1 + m2) / 2
midpoint = midpoint.flatten()

# 分隔線的法向量是 w，所以分隔線方向是垂直於 w 的向量
# 如果 w = [a, b]，則垂直向量 = [-b, a]
w_perp = np.array([-w[1], w[0]])

plt.figure(dpi=288)

plt.plot(X1[:, 0], X1[:,1], 'r.')
plt.plot(X2[:, 0], X2[:,1], 'g.')

# write you code here
# 繪製分隔線
t = np.linspace(-5, 5, 100)
line_x = midpoint[0] + t * w_perp[0]
line_y = midpoint[1] + t * w_perp[1]
plt.plot(line_x, line_y, 'b-', linewidth=2, label='Decision Boundary')

# 繪製投影方向（箭頭）
arrow_scale = 2.0
plt.arrow(midpoint[0], midpoint[1], w[0] * arrow_scale, w[1] * arrow_scale, 
          head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=2)

plt.legend()
plt.axis('equal')  
plt.show()
