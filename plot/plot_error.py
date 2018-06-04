import matplotlib.pyplot as plt
import json
from scipy.interpolate import spline
import numpy as np
import sys

path = './data/lfa/20180604-153743/lfa.error.json'

with open(path, 'r') as file:
    data = json.load(file)

x = np.asarray([d['error'] for d in data])
y = np.asarray(list(range(0, len(x))))

# plt.boxplot(x, 0, 'rs', 0)
plt.figure(figsize=(11.69, 8.27))
plt.plot(x)
plt.title(
    'Error Over Time (Training) - CartPole - LFA', fontsize=18)
plt.xlabel('LFA Update Step', fontsize=18)
plt.ylabel('Error', fontsize=18)

plt.show()
