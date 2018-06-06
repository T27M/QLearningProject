import numpy as np
import matplotlib.pyplot as plt

# 11085
# 31494
# 42701

# 44228
# 204447
# 407625

objects = ('100', '500', '1,000')
y_pos = np.arange(len(objects))
performance = [44228, 204447, 407625]

plt.figure(figsize=(11.69, 8.27))

plt.bar(0, 44228, align='center')
plt.bar(1, 204447, align='center')
plt.bar(2, 407625, align='center')

plt.xticks(y_pos, objects)
plt.xlabel('Episodes', fontsize=18)
plt.ylabel('Number of Key States', fontsize=18)
plt.title('Pacman - Average Key States per Episodes', fontsize=18)


plt.show()
