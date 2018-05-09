import json
import pickle
from qlearning.qtableentry import QTableEntry


class QTable(object):
    def __init__(self, actions):
        self.__q_table = {}
        self.__actions = actions

    def load_q_table(self, file_name):
        with open(file_name, 'r') as file:
            self.__q_table = file.read(pickle.loads(file))

    def save_q_table(self):
        print('Saving QTable...')

        with open('./data/qtable.json', 'wb') as file:
            file.write(pickle.dumps(self.__q_table))

    def get_q_table_len(self):
        return len(self.__q_table)

    def get_q_table_values(self, state):
        """ Gets the q_table for a given state

        Arguments:
            state {list} -- the state vector

        Returns:
            dict -- q table values
        """
        return self.__get_q_table_entry(state).get_q_values()

    def get_q_table_value(self, state, action):
        """ Gets a q_table value for a given state and action

        Arguments:
            state {list} -- the state vector

        Returns:
            dict -- q table values
        """
        return self.__get_q_table_entry(state).get_q_value(action)

    def update_q_table_value(self, state, action, new_value):
        """ Updates the qvalue of an action state pair

        Arguments:
            state {list} -- state vector
            action {string} -- name of the action taken
            new_value {int} -- the new q value
        """
        # Search for the entry in the qtable
        self.__get_q_table_entry(state).set_q_value(action, new_value)

    def get_max_q_table_value(self, state):
        """ Get max qvalue for a given state

        Arguments:
            state {list} -- state vector

        Returns:
            float -- the qvalue
        """
        return self.__get_q_table_entry(state).get_q_value_max()

    def __get_q_table_entry(self, state):
        # Convert state to temporary entry to use for searching
        q_table_entry = QTableEntry(state, actions=self.__actions)
        key = q_table_entry.get_state_str()

        # Check if entry exists, if not create
        if(key not in self.__q_table):
            self.__q_table[key] = q_table_entry

        return self.__q_table[key]
