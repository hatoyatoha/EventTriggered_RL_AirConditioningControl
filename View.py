# 資料グラフ出力用
# python==3.7.16
from pathlib import Path
import random
import numpy as np  #numpy==1.21.4
import pandas as pd  #pandas==1.2.4
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns  #seaborn==0.11.1
from MDP_train6 import train as mdp_train
from SMDP_train import train as smdp_train
from MIX_train import train as mix_train

def view(mdp_threshold_high_list, mdp_threshold_low_list, mdp_std_list, mdp_j_list, mdp_actor_loss_list, mdp_critic_loss_list, mdp_time_list, smdp_threshold_high_list, smdp_threshold_low_list, smdp_std_list, smdp_j_list, smdp_actor_loss_list, smdp_critic_loss_list, smdp_time_list, mix_threshold_high_list, mix_threshold_low_list, mix_std_list, mix_j_list, mix_actor_loss_list, mix_critic_loss_list, mix_time_list):
    fig1 = plt.figure()
    ax1 = fig1.subplots()
    ax2 = ax1.twinx()
    sns.lineplot(x=mix_time_list, y=mix_threshold_high_list, ax=ax1, color="#FF4B00", ci=None, linewidth=2)
    sns.lineplot(x=mix_time_list, y=mix_threshold_low_list, ax=ax1, color="#005AFF", ci=None, linewidth=2)
    sns.lineplot(x=mix_time_list, y=mix_std_list, ax=ax2, color="#03AF7A", ci=None)
    ax1.set_xlabel("time(hour)")  #x軸ラベル
    ax1.set_ylabel("situonn （℃）")  #y1軸ラベル
    ax2.set_ylabel("標準偏差")  #y2軸ラベル
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    sns.lineplot(x=mdp_time_list, y=mdp_j_list, color="#03AF7A", alpha = 0.8, ci=None)
    sns.lineplot(x=smdp_time_list, y=smdp_j_list, color="#005AFF", ci=None, linewidth=2)
    sns.lineplot(x=mix_time_list, y=mix_j_list, color="#FF4B00", ci=None, linewidth=2)
    plt.xlabel("time(hour)")  #x軸ラベル
    plt.ylabel("平均報酬")  #y軸ラベル
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    mdp_threshold_high_list, mdp_threshold_low_list, mdp_std_list, mdp_j_list, mdp_actor_loss_list, mdp_critic_loss_list, mdp_time_list = mdp_train()
    smdp_threshold_high_list, smdp_threshold_low_list, smdp_std_list, smdp_j_list, smdp_actor_loss_list, smdp_critic_loss_list, smdp_time_list = smdp_train()
    mix_threshold_high_list, mix_threshold_low_list, mix_std_list, mix_j_list, mix_actor_loss_list, mix_critic_loss_list, mix_time_list = mix_train()
    view(mdp_threshold_high_list, mdp_threshold_low_list, mdp_std_list, mdp_j_list, mdp_actor_loss_list, mdp_critic_loss_list, mdp_time_list, smdp_threshold_high_list, smdp_threshold_low_list, smdp_std_list, smdp_j_list, smdp_actor_loss_list, smdp_critic_loss_list, smdp_time_list, mix_threshold_high_list, mix_threshold_low_list, mix_std_list, mix_j_list, mix_actor_loss_list, mix_critic_loss_list, mix_time_list)

