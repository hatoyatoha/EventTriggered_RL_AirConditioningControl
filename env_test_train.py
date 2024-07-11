# 環境のコードの動作確認
#python==3.7.16
from pathlib import Path
import random
import numpy as np  #numpy==1.21.4
import pandas as pd  #pandas==1.2.4
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm
import japanize_matplotlib
import seaborn as sns  #seaborn==0.11.1
from tqdm import tqdm
import gym  #gym==0.24.1
from gym.wrappers import RecordVideo
import torch  #torch==1.11.0
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from env import Environment

threshold_high_list = []
threshold_low_list = []
j_list = []
best_j = -100000
hour = 300

for h in np.arange(15.5, 20, 0.2):
    for l in np.arange(10, 14, 0.2):
        environment = Environment()
        in_temp = 15  #室内気温
        out_temp = -10  #外気温
        h_status = 0  #heater:ON=1, OFF=0
        state = np.array([in_temp, out_temp, h_status])
        mean = np.array([h, l])
        j = 0
        j_s = 0
        pre_time = 0
        time = 0
        count = 0

        while mean[1] <= in_temp and in_temp <= mean[0]:  #閾値内なら繰り返し
            next_h_status = h_status  #h_statusは変えない
            time += 1
            step_interval = time - pre_time
            in_temp = environment.step(state, next_h_status, step_interval)[0]
        next_state = environment.step(state, next_h_status, step_interval)  #最初の閾値到達時の状態
        reward = environment.reward(state, next_state, pre_time, time)  #状態，次状態，時間間隔から報酬
        state = next_state
        pre_time = time
        count += 1
        
        while time <= hour*60:  #規定時間(分)に達するまで繰り返し
            if in_temp >= mean[0]:  #OFF閾値以上の場合
                while mean[1] <= in_temp:  #ON閾値以上なら繰り返し
                    next_h_status = 0
                    time += 1
                    step_interval = time - pre_time
                    in_temp = environment.step(state, next_h_status, step_interval)[0]
                next_state = np.array([in_temp, out_temp, next_h_status])  #閾値到達時の状態
                reward = environment.reward(state, next_state, pre_time, time)  #状態，次状態，時間間隔から報酬
                state = next_state
                pre_time = time
                j += reward
                j_s += 0.05*reward/step_interval
                count += 1
                
            elif mean[1] >= in_temp:  #ON閾値以下の場合
                while mean[0] >= in_temp:  #OFF閾値以下なら繰り返し
                    next_h_status = 1
                    time += 1
                    step_interval = time - pre_time
                    in_temp = environment.step(state, next_h_status, step_interval)[0]
                next_state = np.array([in_temp, out_temp, next_h_status])  #閾値到達時の状態
                reward = environment.reward(state, next_state, pre_time, time)  #状態，次状態，時間間隔から報酬
                state = next_state
                pre_time = time
                j += reward
                j_s += 0.05*reward/step_interval
                count += 1
                
        threshold_high_list.append(h)
        threshold_low_list.append(l)
        j_list.append(j/hour)
        
        if best_j < j:
            best_mean = mean
            best_j = j
            best_j_s = j_s
        

print('best', best_mean, best_j/hour, best_j_s)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-8,-3)
ax.plot_trisurf(threshold_high_list, threshold_low_list, j_list, cmap="bwr")
ax.scatter(16.9, 13.2, -8, color='red')
ax.set_xlabel('OFFにする閾値（℃）', size=14)
ax.set_ylabel('ONにする閾値（℃）', size=14)
ax.set_zlabel('平均報酬（unit/hour）', size=14, rotation=90)
plt.show()
