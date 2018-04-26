import numpy as np
from qlearning.qtable import QTable
from core.configbase import ConfigBase
# Loss = ∑(Q-target - Q)²


class QAgent(ConfigBase):
    def __init__(self, config):

        super().__init__(config=config)

        # State table
        self.__QTable = QTable()

        # gamma - encourages rewards sooner > later
        self.__discount_factor = self._config['discount_factor']

        # alpha - how much we update per action: 0 < a < 1
        self.__learning_rate = self._config['learning_rate']

        # epsilon - exploration
        self.__random_action = self._config['random_action']

    def get_q_values(self, state):
        return self.__QTable.get_q_table_values(state)

    def get_q_value(self, state, action):
        return self.__QTable.get_q_table_value(state, action)

    def predict(self, state):
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

    def update_q_table(self, state, new_state, action, reward):
        cur_q_value = self.__QTable.get_q_table_value(state, action)
        new_state_max_q_value = self.__QTable.get_max_q_table_value(new_state)

        # Calculate the new qvalue with the q-learning algorithm
        new_q_value = ((1 - self.__learning_rate) * cur_q_value) + \
            self.__learning_rate * \
            (reward + self.__discount_factor * new_state_max_q_value)

        # Update Q-Table
        self.__QTable.update_q_table_value(state, action, new_q_value)

        # Update Q-Table
        #  ((1 - self.learning_rate) * self.QTable[ob, action]) + self.learning_rate * (reward + self.discount_factor * np.max(self.QTable[ob1, :]))
