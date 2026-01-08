# -*- coding: utf-8 -*-
"""
改寫版本：使用不同風格達到相同視覺結果
作者：Grok 協助改寫（原作者：賴楷崴）
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

# -----------------------------
# 讀取資料
# -----------------------------
# 請將 hw8.csv 放在與此腳本相同目錄，或修改路徑
hw8_csv = pd.read_csv('hw8.csv')  # 建議放在同資料夾
data = hw8_csv.to_numpy(dtype=np.float64)

X = data[:, :2]      # 特徵 x1, x2
y = data[:, 2]       # 標籤 +1 或 -1

# -----------------------------
# 訓練 RBF Kernel SVM
# -----------------------------
clf = SVC(kernel='rbf', C=5.0, gamma='scale')
clf.fit(X, y)

# -----------------------------
# 自訂函式：產生繪圖用網格
# -----------------------------
def make_grid(X, padding=1.0, n_points=500):
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n_points),
        np.linspace(y_min, y_max, n_points)
    )
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]

xx, yy, grid_points = make_grid(X, padding=1.0, n_points=500)

# -----------------------------
# 預測網格點（用於背景著色）
# -----------------------------
Z_pred = clf.predict(grid_points)
Z_pred = Z_pred.reshape(xx.shape)

# 用於畫決策邊界（更精確）：decision function = 0 的等高線
Z_decision = clf.decision_function(grid_points)
Z_decision = Z_decision.reshape(xx.shape)

# -----------------------------
# 繪圖（物件導向風格）
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 6), dpi=288)

# 背景顏色：根據預測類別上色（-1 → 淺綠, +1 → 深綠）
cmap_light = ListedColormap(['#d5f5d5', '#1b5e20'])
ax.contourf(xx, yy, (Z_pred > 0).astype(int), alpha=0.6, cmap=cmap_light)

# 決策邊界（decision function = 0）
ax.contour(xx, yy, Z_decision, levels=[0], colors='black', linewidths=1.5)

# 畫資料點
ax.scatter(X[y ==  1, 0], X[y ==  1, 1], color='red',   label=r'$\omega_1$', edgecolors='k', s=50)
ax.scatter(X[y == -1, 0], X[y == -1, 1], color='blue',  label=r'$\omega_2$', edgecolors='k', s=50)

# 設定
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_aspect('equal', adjustable='box')
ax.legend()
ax.grid(False)

plt.tight_layout()
plt.show()
