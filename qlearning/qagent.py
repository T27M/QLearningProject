import numpy as np
import random
from qlearning.qtable import QTable
from core.configbase import ConfigBase


# Loss = ∑(Q-target - Q)²


class QAgent(ConfigBase):
    def __init__(self, config):

        super().__init__(config=config)

        # Actions that can be taken
        self.__actions = self._config['actions']

        # State table
        self.__QTable = QTable(actions=self.__actions)

        # gamma - encourages rewards sooner > later
        self.__discount_factor = self._config['discount_factor']

        # alpha - how much we update per action: 0 < a < 1
        self.__learning_rate = self._config['learning_rate']

        # epsilon - exploration
        self.__random_action = self._config['random_action']

    def save_q_table(self):
        self.__QTable.save_q_table()

    def load_q_table(self):
        self.__QTable.load_q_table('./data/qtable.json')

        print("Loaded: " + str(self.get_q_table_len()) + " keys")

    def get_q_table_len(self):
        return self.__QTable.get_q_table_len()

    def get_q_values(self, state):
        return self.__QTable.get_q_table_values(state)

    def get_q_value(self, state, action):
        return self.__QTable.get_q_table_value(state, action)

    def predict(self, state):
        return self.__epsilon_greedy_act(state)

    def __epsilon_greedy_act(self, state):
        """ Determines which action is best given the state, or random action based on epsilon

        Arguments:
            state {list} -- state vector

        Returns:
            str -- the action to take
        """
        if self.__random_action > random.random():
            # explore enviroment
            random_action = random.sample(self.__actions, 1)

            return random_action[0]
        else:
            _, action = self.__QTable.get_max_q_table_value(state)

            # greey action
            return action

    def update_q_table(self, state, new_state, action, reward):
        """ Updates the value of a state based on the Q-Learning algorithm
                q(state)[action] = q(state, action) + \
                    alpha * (reward + gamma * np.max(q(next_state)) - q(state, action))

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
        new_q_value = \
            cur_q_value + self.__learning_rate * \
            (reward + self.__discount_factor * new_state_max_q_value - cur_q_value)

        if self._log_output:
            log_line = str(state) + "\t\t| " + str(action) + "\t\t| " + \
                str(reward) + "\t\t| " + str(cur_q_value) + \
                "\t\t| " + str(new_q_value) + "\n"

            with open(self._log_path, 'a') as log:
                log.write(log_line)

                # Update Q-Table
        self.__QTable.update_q_table_value(state, action, new_q_value)
