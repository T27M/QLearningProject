import matplotlib.pyplot as plt
import json
from scipy.interpolate import spline
import numpy as np
import sys
import math

random_baseline = './data/random/qtable/qt.reward.json'
path_train = './data/qtable/0.2_0.9_0.1_10000-train/qt.reward.json'
path_test = './data/qtable/0.2_0.9_0.1_10000-eval/qt.reward.json'

# path_train = '/home/tom/SCP/5000/qt.reward-train.json'
# path_test = '/home/tom/SCP/5000/qt.reward-eval.json'

with open(random_baseline, 'r') as random_file:
    random_data = json.load(random_file)

with open(path_train, 'r') as train_file:
    traiing_data = json.load(train_file)

with open(path_test, 'r') as eval_file:
    eval_data = json.load(eval_file)

random_rewards = np.asarray([d['episode_reward'] for d in random_data])

train_rewards = np.asarray([d['episode_reward'] for d in traiing_data])
eval_rewards = np.asarray([d['episode_reward'] for d in eval_data])

print("Train Avg:" + str(sum(random_rewards) / len(random_rewards)))
print(np.max(random_rewards))

print("Train Avg:" + str(sum(train_rewards) / len(train_rewards)))
print(np.max(train_rewards))

print("Eval Avg:" + str(sum(eval_rewards) / len(eval_rewards)))
print(np.max(eval_rewards))

fig = plt.figure(figsize=(11.69, 8.27))

plt.title(
    'Results - CartPole using Q-Table', fontsize=18)
plt.xlabel('Agent', fontsize=18)
plt.ylabel('Score', fontsize=18)

plt.show()
