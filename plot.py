import matplotlib.pyplot as plt
import json
from scipy.interpolate import spline
import numpy as np
import sys

path = './Cart-Pole-Solved/lfa.reward.json'

with open(path, 'r') as file:
    data = json.load(file)

x = np.asarray([d['episode_reward'] for d in data])
y = np.asarray(list(range(0, len(x))))

# plt.boxplot(x, 0, 'rs', 0)
plt.plot(x)
plt.title('Score Over Time (Training) - CartPole - LFA')
plt.xlabel('Episode')
plt.ylabel('Episode Reward')

plt.show()
