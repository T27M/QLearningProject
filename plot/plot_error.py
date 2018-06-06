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


path = '/home/tom/SCP/lfa/cartpole/0.000001/lfa.error.json'

with open(path, 'r') as file:
    data = json.load(file)

err = np.asarray([d['error'] for d in data])

N = 100
cumsum, moving_aves = [0], []

for i, x in enumerate(err, 1):
    cumsum.append(cumsum[i-1] + x)
    if i >= N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        # can do stuff with moving_ave here
        moving_aves.append(moving_ave)

plt.figure(figsize=(11.69, 8.27))
plt.plot(err, label="Error")
plt.plot(moving_aves, label="Moving Average (per 100 LFA updates)")

plt.title(
    'Error Over Time (Training) - CartPole - LFA', fontsize=18)
plt.xlabel('LFA Update Step', fontsize=18)
plt.ylabel('Error', fontsize=18)

plt.legend(fontsize=18)

episode = '1,000'
lr = '0.000001'
df = '0.9'
ra = '0.1'

textonly(plt.gca(), 'Episodes: ' + episode + '\n' +
         r'$\alpha$:' + lr + '\n' + r'$\gamma$:' + df + '\n' + r'$\epsilon$:' + ra, loc=1)

plt.show()
