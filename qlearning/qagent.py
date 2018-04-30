import numpy as np
import random
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
        return self.__epsilon_greedy_act(state)

    def __epsilon_greedy_act(self, state):
        if self.__random_action > random.random():
            # explore enviroment
            actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            random_action = random.sample(actions, 1)

            return random_action[0]
        else:
            _, action = self.__QTable.get_max_q_table_value(state)

            # greey action
            return action

        # Below adds noise (type? why?)
        # return np.argmax(self.QTable[ob,:] + np.random.randn(1, self.action_space.n) * ( 1. / (i + 1)))
        pass

    def update_q_table(self, state, new_state, action, reward):
        """ Updates the value of a state based on the Q-Learning algorithm
         ((1 - learning_rate) * QTable[s, action]) + learning_rate * (reward + discount_factor * np.max(QTable[s1]))


        Arguments:
            state {list} -- state vector
            new_state {list} -- new state vector
            action {string} -- action taken to move between states
            reward {int} -- reward gained from changing from s to s1
        """
        cur_q_value = self.__QTable.get_q_table_value(state, action)
        new_state_max_q_value, _ \
            = self.__QTable.get_max_q_table_value(new_state)

        # Calculate the new qvalue with the q-learning algorithm
        new_q_value = ((1 - self.__learning_rate) * cur_q_value) + \
            self.__learning_rate * \
            (reward + self.__discount_factor * new_state_max_q_value)

        # Update Q-Table
        self.__QTable.update_q_table_value(state, action, new_q_value)

        # Update Q-Table
