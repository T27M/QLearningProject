import matplotlib.pyplot as plt
import json
from scipy.interpolate import spline
import numpy as np
import sys

path_train = './data/qtable/0.2_0.9_0.1_10000-train/qt.reward.json'
path_test = './data/qtable/0.2_0.9_0.1_10000-eval/qt.reward.json'

# path_train = '/home/tom/SCP/5000/qt.reward-train.json'
# path_test = '/home/tom/SCP/5000/qt.reward-eval.json'

with open(path_train, 'r') as train_file:
    traiing_data = json.load(train_file)

with open(path_test, 'r') as eval_file:
    eval_data = json.load(eval_file)

train_rewards = np.asarray([d['episode_reward'] for d in traiing_data])
eval_rewards = np.asarray([d['episode_reward'] for d in eval_data])

fig = plt.figure(figsize=(11.69, 8.27))

ax = fig.add_subplot(111)

bp = ax.boxplot([train_rewards, eval_rewards], showfliers=False)

ax.set_xticklabels(['Training', 'Evaluation'], fontsize=18)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)
plt.title(
    'Score Over Time - CartPole - Q-Table', fontsize=18)
plt.xlabel('Agent', fontsize=18)
plt.ylabel('Score', fontsize=18)

plt.show()
