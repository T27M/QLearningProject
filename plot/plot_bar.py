import numpy as np
import matplotlib.pyplot as plt

# 11085
# 31494
# 42701

objects = ('10,000', '50,000', '100,000')
y_pos = np.arange(len(objects))
performance = [11085, 31494, 42701]

plt.figure(figsize=(11.69, 8.27))

plt.bar(0, 11085, align='center')
plt.bar(1, 31494, align='center')
plt.bar(2, 42701, align='center')

plt.xticks(y_pos, objects)
plt.xlabel('Episodes', fontsize=18)
plt.ylabel('Number of Key States', fontsize=18)
plt.title('CartPole - Average Key States per Episodes', fontsize=18)


plt.show()
