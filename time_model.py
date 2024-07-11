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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#クリティックのNNによる状態価値
class CriticNet(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):

        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)  #入力層，隠れ層
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)  #隠れ層，隠れ層
        self.linear3 = nn.Linear(hidden_dim, output_dim)  #隠れ層，出力層

    def forward(self, state):  #データstateから価値を出力
        x = state
        x = self.linear1(x)
        x = F.relu(x)  #linearからrelu
        x = self.linear3(x)  #linear
        return x

#アクターの行動を分布から出力
class Actor():

    def sample(self, mean, std):  #平均と対数偏差から正規分布を作成し，行動と平均（行動）を出力
        done = False
        normal = Normal(mean, std)
        h = torch.mean(mean).item()
        d = mean[0][0].item() - h
        while not done:
            action = normal.rsample()  #正規分布からサンプリングする
            if mean[0][0].item()+d/2 > action[0][0].item() and mean[0][0].item()-d/2 < action[0][0].item() and mean[0][1].item()+d/2 > action[0][1].item() and mean[0][1].item()-d/2 < action[0][1].item():  #離れすぎの値はやり直し
                done = True
        return action, mean, std

class ActorCriticModel(object):

    def __init__(self, state_dim, args, device):

        self.alpha = args['alpha']  #学習率
        self.device = device  #cpu

        self.actor_net = Actor()  #アクター
        self.critic_net = CriticNet(input_dim=state_dim, output_dim=1, hidden_dim=args['hiden_dim']).to(self.device)  #クリティックネットワーク
        self.critic_optim = optim.Adam(self.critic_net.parameters())  #CriticNetのパラメータを最適化するために、Adamオプティマイザを使用

    def select_action(self, mean, std, evaluate=False):  #状態から行動選択
        mean = torch.FloatTensor(mean).unsqueeze(0).to(self.device)  #平均を表すテンソルを作成
        std = torch.FloatTensor(std).unsqueeze(0).to(self.device)  #標準偏差を表すテンソルを作成
        if not evaluate:  #学習時
            action, _, std = self.actor_net.sample(mean, std)  #確率分布をかませた行動
        else:  #評価時
            _, action, std = self.actor_net.sample(mean, std)  #決定的方策
        return action.detach().numpy().reshape(-1), std.detach().numpy().reshape(-1)  #行動,標準偏差をnumpy配列に変換し、次元を整形

    def update_parameters(self, memory, batch_size, j, step_interval, i):  #パラメータを更新し，損失関数を返す

        state_batch, action_batch, reward_batch, next_state_batch, norm_action = memory.sample(batch_size=batch_size)  #ReplayMemoryからバッチサイズ分の経験をサンプリングし、状態、行動、報酬、次の状態、マスクを取得

        state_batch = torch.FloatTensor(state_batch)  #テンソルに変換
        next_state_batch = torch.FloatTensor(next_state_batch)
        action_mean = action_batch[0]
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        mean = norm_action[0][0:2]
        std = np.array([norm_action[0][2]])

        with torch.no_grad():  #テンソルから勾配情報を消す
            next_v_values = reward_batch - j*step_interval + self.critic_net(next_state_batch)  #V'の更新 推定値V=r-j+V'

        v_values = self.critic_net(state_batch)  #状態と行動からV
        actor_loss = next_v_values - v_values  #VとV'からアクターの損失関数  delta
        actor_loss = actor_loss.detach().numpy()[0][0]
        critic_loss = F.mse_loss(v_values, next_v_values)  #VとV'からクリティックの損失関数  delta

        j = j + 0.05*actor_loss.item()/step_interval

        #アクター更新
        if i == 0:  #OFF閾値を更新
            mean_diff_logpi_high = (action_mean[0] - mean[0])/(std[0]**2)
            mean_diff_logpi = np.array([mean_diff_logpi_high, 0])
            std_diff_logpi_high = (((action_mean[0] - mean[0])**2) - (std[0]**2))/(std[0]**3)
            std_diff_logpi = np.array([std_diff_logpi_high])
        
        elif i == 1:  #ON閾値を更新
            mean_diff_logpi_low = (action_mean[1] - mean[1])/(std[0]**2) 
            mean_diff_logpi = np.array([0, mean_diff_logpi_low])
            std_diff_logpi_low = (((action_mean[1] - mean[1])**2) - (std[0]**2))/(std[0]**3)
            std_diff_logpi = np.array([std_diff_logpi_low])

        mean = mean - self.alpha*actor_loss*mean_diff_logpi
        std = std - 0.05*actor_loss*std_diff_logpi
        std = np.clip(std, 0.2, 5)  #標準偏差をクリッピング

        #クリティック更新
        self.critic_optim.zero_grad()  #勾配を0に初期化
        critic_loss.backward()  #誤差逆伝播法
        self.critic_optim.step()  #学習

        return mean, std, j, actor_loss, critic_loss.tolist()  #テンソルからクリティック損失関数とアクター損失関数の要素を取得

#過去の経験を再利用
class ReplayMemory:

    def __init__(self, memory_size):
        self.memory_size = memory_size  #バッファのサイズ
        self.buffer = []  #経験を格納するためのリスト
        self.position = 0  #バッファ内の次に書き込む場所

    def push(self, state, action, reward, next_state, norm_action):  #状態，行動，報酬，次状態，終端(1)終端でない(0)
        self.buffer.clear()
        self.buffer.append((state, action, reward, next_state, norm_action))

    def sample(self, batch_size):  #バッファからランダムにバッチを抽出
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, norm_action = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, norm_action

    def __len__(self):  #バッファの要素数
        return len(self.buffer)
