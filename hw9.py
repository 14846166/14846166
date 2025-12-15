# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:46:50 2021

@author: htchen
"""
#14846166 賴楷崴
#from IPython import get_ipython
#get_ipython().run_line_magic('reset', '-sf')
import math
import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt
import pandas as pd
figdpi = 400

hw9_csv = pd.read_csv(r'C:\Users\ASUS\Downloads\OneDrive_1_2025-10-31\hw9(in).csv').to_numpy(dtype = np.float64)
t = hw9_csv[:, 0] # 時間
flow_velocity = hw9_csv[:, 1] # 氣體流速
plt.figure(dpi=figdpi)
plt.plot(t, flow_velocity, 'r')
plt.title('Gas Flow Velocity')
plt.xlabel('time in seconds')
plt.ylabel('ml/sec')
plt.show()

# Integrating the gas flow velocity yields the net flow
net_vol = np.cumsum(flow_velocity) * 0.01
plt.figure(dpi=figdpi)
plt.plot(t, net_vol, 'r')
plt.title('Gas Net Flow')
plt.xlabel('time in seconds')
plt.ylabel('ml')
plt.show()

A = np.zeros((len(t), 3))
A[:, 0] = 1
A[:, 1] = t
A[:, 2] = t * t
y = net_vol
a = la.inv(A.T @ A) @ A.T @ y
trend_curve = a[0] + a[1] * t + a[2] * t * t

net_vol_corrected = net_vol - trend_curve

plt.figure(dpi=figdpi)
plt.plot(t, net_vol_corrected, 'b')
plt.title('Gas Net Flow (Corrected - Quadratic Detrending)')
plt.xlabel('time in seconds')
plt.ylabel('ml')
plt.savefig('Gas_Net_Flow_Corrected.png')
plt.show()
plt.close()

