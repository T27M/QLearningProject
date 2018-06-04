import matplotlib.pyplot as plt
import json
from scipy.interpolate import spline
import numpy as np
import sys

path_train = './data/qtable/20180604-173706/qt.reward.json'
path_test = './data/qtable/20180604-173801/qt.reward.json'

with open(path_train, 'r') as train_file:
    traiing_data = json.load(train_file)

with open(path_test, 'r') as eval_file:
    eval_data = json.load(eval_file)

train_rewards = np.asarray([d['episode_reward'] for d in traiing_data])
eval_rewards = np.asarray([d['episode_reward'] for d in eval_data])

fig = plt.figure(figsize=(11.69, 8.27))

ax = fig.add_subplot(111)

bp = ax.boxplot([train_rewards, eval_rewards])

ax.set_xticklabels(['Training', 'Evaluation'], fontsize=18)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)
plt.title(
    'Score Over Time (Evaluation) - CartPole - Q-Table', fontsize=18)
plt.xlabel('Agent', fontsize=18)
plt.ylabel('Score', fontsize=18)

plt.show()
