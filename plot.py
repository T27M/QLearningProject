import matplotlib.pyplot as plt
import json
from scipy.interpolate import spline
import numpy as np
import sys

# path = './data/lfa/cart-pole-opt-500/lfa.reward.json'
path = './Cart-Pole-Solved/lfa.reward.json'

with open(path, 'r') as file:
    data = json.load(file)

x = np.asarray([d['episode_reward'] for d in data])
y = np.asarray(list(range(0, len(x))))

# plt.boxplot(x, 0, 'rs', 0)
plt.figure(figsize=(11.69, 8.27))
plt.plot(x)
plt.title(
    'Score Over Time (Training) - CartPole - LFA', fontsize=18)
plt.xlabel('Episode', fontsize=18)
plt.ylabel('Episode Reward', fontsize=18)

plt.show()
