import numpy as np
from qlearning.qtable import QTable

# Holds a Q-Table with state to action pairs

# Allows the use of a hashing function to

# Process to add
# Take obs and hash
# Check if hash exists in qtable
#

# Loss = ∑(Q-target - Q)²


class QAgent(object):
    def __init__(self, config):
        # State table
        self.__QTable = QTable()

        # gamma - encourages rewards sooner > later
        self.__discount_factor = config['discount_factor']

        # alpha - how much we update per action: 0 < a < 1
        self.__learning_rate = config['learning_rate']

        # epsilon - exploration
        self.__random_action = config['random_action']

    def predict(self, observation):
        pass

    def __greedy_act(self, ob, i):
        # if self.random_action > random.random():
            # explore enviroment
            # return self.action_space.sample()
        # else:
            # greey action
            # return np.argmax(self.QTable[ob,:])

        # return np.argmax(self.QTable[ob,:] + np.random.randn(1, self.action_space.n) * ( 1. / (i + 1)))
        pass

    def __update_q_table(self, ob, ob1, action, reward):
        # Update Q-Table
        # self.QTable[ob, action] = ((1 - self.learning_rate) * self.QTable[ob, action]) + self.learning_rate * (reward + self.discount_factor * np.max(self.QTable[ob1, :]))

        pass
