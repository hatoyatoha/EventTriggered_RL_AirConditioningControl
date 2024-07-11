#環境
#python==3.7.16
from pathlib import Path
import random
import numpy as np  #numpy==1.21.4
import pandas as pd  #pandas==1.2.4
import matplotlib.pyplot as plt
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
from scipy import integrate

class Environment(object):
    def __init__(self):
        self.d_temp = 15  #目標温度
        self.r_sw = -0.8  #ヒーターONOFFペナルティ
        self.r_e = -1.2/3600  #エネルギー消費ペナルティ
        self.r_c = -0.6/3600  #目標温度との差のペナルティ-1.2/3600
        
        self.C = 3000*1000  #熱容量
        self.K = 325  #熱伝導率
        self.Q = 13*1000  #ヒーター能力


    #状態遷移関数
    def step(self, state, next_h_status, step_interval):
        
        in_temp = state.tolist()[0]
        out_temp = state.tolist()[1]
        h_status = state.tolist()[2]

        a = in_temp - out_temp - (self.Q/self.K)*next_h_status
        next_in_temp = a*np.exp(-self.K*step_interval*60/self.C) + out_temp + (self.Q/self.K)*next_h_status
        return np.array([next_in_temp, out_temp, next_h_status])

    #報酬関数
    def reward(self, state, next_state, pre_time, time):
        in_temp = state.tolist()[0]
        out_temp = state.tolist()[1]
        h_status = state.tolist()[2]
        next_in_temp = next_state.tolist()[0]
        next_h_status = next_state.tolist()[2]
        step_interval = time - pre_time
        if h_status == next_h_status:
            r_sw = 0
        else:
            r_sw = self.r_sw
        a = (in_temp - out_temp - (self.Q/self.K)*next_h_status)*np.exp(self.K*pre_time*60/self.C)
        def f(x):
            return (out_temp + (self.Q/self.K)*next_h_status + a*np.exp(-self.K*x/self.C) - self.d_temp)**2
        ans, _ = integrate.quad(f, pre_time*60, time*60)
        reward = r_sw + self.r_e*next_h_status*step_interval*60 + self.r_c*ans

        return reward  #unit
