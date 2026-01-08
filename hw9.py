# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:46:50 2021

@author: htchen
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# 1. 讀取資料（建議將 hw9(in).csv 放在與此腳本相同資料夾）
# -----------------------------
data = pd.read_csv('hw9(in).csv').to_numpy(dtype=np.float64)

t = data[:, 0]              # 時間 (秒)
flow_velocity = data[:, 1]  # 氣體流速 (ml/sec)

# 計算時間間隔（假設資料均勻採樣，從相鄰點推斷）
dt = t[1] - t[0]            # 應該是 0.01 秒
print(f"偵測到時間間隔 dt = {dt} 秒")  # 可確認是否正確

# -----------------------------
# 2. 原始氣體流速圖
# -----------------------------
fig1, ax1 = plt.subplots(dpi=400)
ax1.plot(t, flow_velocity, color='red', linewidth=1.5)
ax1.set_title('Gas Flow Velocity')
ax1.set_xlabel('time in seconds')
ax1.set_ylabel('ml/sec')
ax1.grid(False)
plt.tight_layout()
plt.show()

# -----------------------------
# 3. 淨流量（對流速積分）
# -----------------------------
# 使用累積積分：net_vol[i] = sum(flow_velocity[0 to i]) * dt
net_vol = np.cumsum(flow_velocity) * dt

fig2, ax2 = plt.subplots(dpi=400)
ax2.plot(t, net_vol, color='red', linewidth=1.5)
ax2.set_title('Gas Net Flow')
ax2.set_xlabel('time in seconds')
ax2.set_ylabel('ml')
ax2.grid(False)
plt.tight_layout()
plt.show()

# -----------------------------
# 4. 二次趨勢擬合並去除（Detrending）
# -----------------------------
# 使用 np.polyfit 直接擬合二次多項式：y = a2*t^2 + a1*t + a0
# degree=2 表示二次
coefficients = np.polyfit(t, net_vol, deg=2)
a2, a1, a0 = coefficients

# 計算趨勢曲線
trend_curve = np.polyval(coefficients, t)  # 等價於 a0 + a1*t + a2*t^2

# 去除趨勢後的淨流量
net_vol_corrected = net_vol - trend_curve

# -----------------------------
# 5. 繪製去除趨勢後的圖並儲存
# -----------------------------
fig3, ax3 = plt.subplots(dpi=400)
ax3.plot(t, net_vol_corrected, color='blue', linewidth=1.5)
ax3.set_title('Gas Net Flow (Corrected - Quadratic Detrending)')
ax3.set_xlabel('time in seconds')
ax3.set_ylabel('ml')
ax3.grid(False)
plt.tight_layout()

# 儲存圖片（與原程式相同檔名）
plt.savefig('Gas_Net_Flow_Corrected.png', dpi=400)
plt.show()

# 關閉圖形（好習慣，避免記憶體累積）
plt.close('all')
