import matplotlib.pyplot as plt
import json
from scipy.interpolate import spline
import numpy as np
import sys
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText


def textonly(ax, txt, fontsize=14, loc=2, *args, **kwargs):
    at = AnchoredText(txt,
                      prop=dict(size=fontsize),
                      frameon=True,
                      loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    at.patch.set_fill(False)
    ax.add_artist(at)
    return at


# random_baseline = './data/random/lfa/lfa.reward.json'

random_baseline = './data/random/qtable/pacman-500/qt.reward.json'

path_train = '/home/tom/SCP/qtable/pacman/02_09_500/qt.reward-train.json'
path_test = '/home/tom/SCP/qtable/pacman/02_09_500/qt.reward-eval.json'

with open(random_baseline, 'r') as random_file:
    random_data = json.load(random_file)

with open(path_train, 'r') as train_file:
    traiing_data = json.load(train_file)

with open(path_test, 'r') as eval_file:
    eval_data = json.load(eval_file)

random_rewards = np.asarray([d['episode_reward'] for d in random_data])

train_rewards = np.asarray([d['episode_reward'] for d in traiing_data])
eval_rewards = np.asarray([d['episode_reward'] for d in eval_data])

print("Random Avg:" + str(sum(random_rewards) / len(random_rewards)))
print(np.max(random_rewards))

print('\n')

train_len = len(train_rewards)
eval_len = len(eval_rewards)

if train_len != eval_len:
    print(train_len)
    print(eval_len)

    input('INVALID DATA? - Are you want to plot this (T=500,E=100)')

train_avg = sum(train_rewards) / train_len
train_max = np.max(train_rewards)

eval_avg = sum(eval_rewards) / eval_len
eval_max = np.max(eval_rewards)

print("Train Avg:" + str(train_avg))
print('Episodes:' + str(train_len))
print('Max Train:' + str(train_max))

print('\n')

print("Eval Avg:" + str(eval_avg))
print('Episodes:' + str(len(eval_rewards)))
print('Max Eval:' + str(eval_max))

fig = plt.figure(figsize=(11.69, 8.27))

ax = fig.add_subplot(111)


bp = ax.boxplot([random_rewards, train_rewards, eval_rewards],
                showfliers=True,  # outliers
                notch=False,  # notch shape
                vert=True,   # vertical box aligmnent
                patch_artist=False,  # fill with color
                sym='+')

ax.set_xticklabels(['Random Agent', 'Q-Table Training \n (1000 episode)',
                    'Q-Table Evaluation'], fontsize=18)


ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)

episode = str(eval_len)
lr = '0.2'
df = '0.9'
ra = '0.1'

textonly(plt.gca(), 'Evaluation Episodes: ' + episode + '\n' +
         r'$\alpha$:' + lr + '\n' + r'$\gamma$:' + df + '\n' + r'$\epsilon$:' + ra, loc=1)

plt.title(
    'Results - Pacman using Q-Table', fontsize=18)
plt.xlabel('Agent', fontsize=18)
plt.ylabel('Score', fontsize=18)

plt.show()
