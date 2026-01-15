# actor:policy gradient, critic:一層NN
# python==3.7.16
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
from event_model import CriticNet
from event_model import Actor
from event_model import ActorCriticModel
from event_model import ReplayMemory
from env import Environment

args = {
    'alpha': 0.05,
    'hiden_dim': 2,
    'memory_size': 1,
    'hour': 300, 
    'threshold_high': 21,
    'threshold_low': 9,
    'standard_deviation': 1
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_state_space=2  #状態空間2次元

def train():
    agent = ActorCriticModel(
        state_dim=env_state_space, args=args, device=device
    )  #状態空間Box，係数，cpu
    memory = ReplayMemory(args['memory_size'])  #メモリのサイズ
    environment = Environment()

    time_list = []
    threshold_high_list = []  #高い閾値
    threshold_low_list = []  #低い閾値
    std_list = []
    j_list = []
    actor_loss_list = []
    critic_loss_list = []

    time = 0
    pre_time = 0
    j = torch.tensor([[0]])

    #初期状態
    in_temp = 13  #室内気温
    out_temp = -10  #外気温
    h_status = 1  #heater:ON=1, OFF=0
    state = np.array([in_temp, out_temp, h_status])
    mean = np.array([args['threshold_high'], args['threshold_low']])
    std = np.array([args['standard_deviation']])
    print('time', time, 'mean', mean, 'std', std, 'intemp', state[0], 'status', state[2])

    threshold_high_list.append(mean[0])
    threshold_low_list.append(mean[1])
    std_list.append(std[0])
    j_list.append(j.tolist()[0][0]*60)
    actor_loss_list.append(0)
    critic_loss_list.append(0)
    time_list.append(round(time/60, 1))
    
    while mean[1] <= in_temp and in_temp <= mean[0]:  #閾値内なら繰り返し
        next_h_status = h_status  #h_statusは変えない
        time += 1
        step_interval = time - pre_time
        in_temp = environment.step(state, next_h_status, step_interval)[0]
    
    next_state = environment.step(state, next_h_status, step_interval)  #最初の閾値到達時の状態
    reward = environment.reward(state, next_state, pre_time, time)  #状態，次状態，時間間隔から報酬
    state = next_state
    action = mean
    pre_time = time
    print('time', time, 'mean', mean, 'std', std, 'intemp', state[0], 'status', state[2])
        
    while time <= args['hour']*60:  #規定時間(分)に達するまで繰り返し
        if in_temp >= action[0]:  #OFF閾値以上の場合
            action, _ = agent.select_action(mean, std)  #確率分布をかませた行動
            decisive_action, std = agent.select_action(mean, std, evaluate=True)  #平均と標準偏差
            threshold_high_list.append(decisive_action[0])
            threshold_low_list.append(decisive_action[1])
            std_list.append(std[0])
            j_list.append(j.tolist()[0][0]*60)
            time_list.append(round(time/60, 1))
            
            while action[1] <= in_temp:  #ON閾値以上なら繰り返し
                next_h_status = 0
                time += 1
                step_interval = time - pre_time
                in_temp = environment.step(state, next_h_status, step_interval)[0]
                
            f_state = np.array([0, 1])
            f_next_state = np.array([1, 0])
            next_state = np.array([in_temp, out_temp, next_h_status])  #閾値到達時の状態
            reward = environment.reward(state, next_state, pre_time, time)  #状態，次状態，時間間隔から報酬
            norm_action = np.concatenate([decisive_action, std])  #平均と標準偏差のndarray
            memory.push(state=f_state, action=action, reward=reward, next_state=f_next_state, norm_action=norm_action)  #メモリに追加
            mean, std, j, actor_loss, critic_loss = agent.update_parameters(memory, 1, j, step_interval, 1)  #パラメータの更新
            actor_loss_list.append(actor_loss)
            critic_loss_list.append(critic_loss)
            state = next_state
            pre_time = time
        
        elif action[1] >= in_temp:  #ON閾値以下の場合
            action, _ = agent.select_action(mean, std)  #確率分布をかませた行動
            decisive_action, std = agent.select_action(mean, std, evaluate=True)  #平均と標準偏差
            threshold_high_list.append(decisive_action[0])
            threshold_low_list.append(decisive_action[1])
            std_list.append(std[0])
            j_list.append(j.tolist()[0][0]*60)
            time_list.append(round(time/60, 1))
            
            while action[0] >= in_temp:  #OFF閾値以下なら繰り返し
                next_h_status = 1
                time += 1
                step_interval = time - pre_time
                in_temp = environment.step(state, next_h_status, step_interval)[0]

            f_state = np.array([1, 0])
            f_next_state = np.array([0, 1])
            next_state = np.array([in_temp, out_temp, next_h_status])  #閾値到達時の状態
            reward = environment.reward(state, next_state, pre_time, time)  #状態，次状態，時間間隔から報酬
            norm_action = np.concatenate([decisive_action, std])  #平均と標準偏差のndarray
            memory.push(state=f_state, action=action, reward=reward, next_state=f_next_state, norm_action=norm_action)  #メモリに追加
            mean, std, j, actor_loss, critic_loss = agent.update_parameters(memory, 1, j, step_interval, 0)  #パラメータの更新
            actor_loss_list.append(actor_loss)
            critic_loss_list.append(critic_loss)
            state = next_state
            pre_time = time
        
        print('time', time, 'mean', mean, 'std', std, 'intemp', state[0], 'status', state[2], 'j', j, 'step_interval', step_interval)
        
    return threshold_high_list, threshold_low_list, std_list, j_list, actor_loss_list, critic_loss_list, time_list

def view(threshold_high_list, threshold_low_list, std_list, j_list, actor_loss_list, critic_loss_list, time_list):
    dataframe = pd.DataFrame(data={'threshold_high_list': threshold_high_list, 'threshold_low_list': threshold_low_list, 'std_list': std_list, 'j_list': j_list, 'actor_loss': actor_loss_list, 'critic_loss': critic_loss_list, 'time_list': time_list})
    fig1 = plt.figure()
    ax1 = fig1.subplots()
    ax2 = ax1.twinx()
    sns.lineplot(x="time_list", y="threshold_high_list", data=dataframe, ax=ax1, color="red", ci=None)
    sns.lineplot(x="time_list", y="threshold_low_list", data=dataframe, ax=ax1, color="blue", ci=None)
    sns.lineplot(x="time_list", y="std_list", data=dataframe, ax=ax2, color="green", ci=None)
    # sns.lineplot(x="time_list", y="threshold_high_list", data=dataframe, ax=ax1, color="red", label="$T_{OFF}$", ci=None)
    # sns.lineplot(x="time_list", y="threshold_low_list", data=dataframe, ax=ax1, color="blue", label="$T_{ON}$", ci=None)
    # sns.lineplot(x="time_list", y="std_list", data=dataframe, ax=ax2, color="green",label="標準偏差", ci=None)
    # handler1, label1 = ax1.get_legend_handles_labels()
    # handler2, label2 = ax2.get_legend_handles_labels()
    # ax1で凡例をまとめて表示
    # ax1.legend(handler1 + handler2, label1 + label2, loc="upper right")
    # # ax2の凡例は削除
    # ax2.get_legend().remove()
    ax1.set_xlabel("time(hour)")  #x軸ラベル
    ax1.set_ylabel("situonn （℃）")  #y1軸ラベル
    ax2.set_ylabel("標準偏差")  #y2軸ラベル
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    sns.lineplot(x="time_list", y="j_list", data=dataframe, color="blue", ci=None)
    plt.xlabel("time(hour)")  #x軸ラベル
    plt.ylabel("平均報酬")  #y軸ラベル
    plt.tight_layout()
    plt.show()
    
    # plt.figure()
    # sns.lineplot(x="time_list", y="actor_loss", data=dataframe, color="blue", label="actor_loss", ci=None)
    # # sns.lineplot(x="time_list", y="critic_loss", data=dataframe, color="red",label="critic_loss", ci=None)
    # plt.xlabel("time(hour)")  #x軸ラベル
    # plt.ylabel("loss")  #y軸ラベル
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    threshold_high_list, threshold_low_list, std_list, j_list, actor_loss_list, critic_loss_list, time_list = train()
    view(threshold_high_list, threshold_low_list, std_list, j_list, actor_loss_list, critic_loss_list, time_list)


