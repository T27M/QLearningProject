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
    ax.add_artist(at)
    return at


path_train = '/home/tom/SCP/lfa/cartpole/0.0001/lfa.reward-train.json'
path_test = '/home/tom/SCP/lfa/cartpole/0.0001/lfa.reward-eval.json'

with open(path_train, 'r') as train_file:
    traiing_data = json.load(train_file)

with open(path_test, 'r') as eval_file:
    eval_data = json.load(eval_file)

train_rewards = np.asarray([d['episode_reward'] for d in traiing_data])
eval_rewards = np.asarray([d['episode_reward'] for d in eval_data])

train_len = len(train_rewards)
eval_len = len(eval_rewards)

if train_len != eval_len:
    print('INVALID DATA')
    sys.exit()

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

N = 100
cumsum, moving_aves = [0], []

for i, x in enumerate(train_rewards, 1):
    cumsum.append(cumsum[i-1] + x)
    if i >= N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        # can do stuff with moving_ave here
        moving_aves.append(moving_ave)

plt.plot(train_rewards, label="Agent Score")
plt.plot(moving_aves, label="Moving Average (per 100 Episode)")

plt.legend(fontsize=18)

episode = str(eval_len)
lr = '0.0001'
df = '0.9'
ra = '0.1'

textonly(plt.gca(), 'Episodes: ' + episode + '\n' +
         r'$\alpha$:' + lr + '\n' + r'$\gamma$:' + df + '\n' + r'$\epsilon$:' + ra, loc=2)

plt.title(
    'CartPole (Training) - LFA', fontsize=18)
plt.xlabel('Episode', fontsize=18)
plt.ylabel('Score', fontsize=18)

plt.show()
