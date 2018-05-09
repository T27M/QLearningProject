import random


class QTableEntry(object):
    def __init__(self, state, actions):
        self.__state_str = str(state)

        self.__state = state

        self.__q_values = {}

        for action in actions:
            self.__q_values[action] = 0.0

    # def get_hash(self):
    #     return self.__hash

    def get_state(self):
        return self.__state

    def get_state_str(self):
        return self.__state_str

    # def compare_hash(self, other):
    #     return self.__hash == other.get_hash()

    def compare_state(self, other):
        return self.__state_str == other.get_state_str()

    def get_q_values(self):
        return self.__q_values

    def get_q_value(self, action):
        return self.__q_values[action]

    def set_q_value(self, action, value):
        self.__q_values[action] = value

    def get_q_value_max(self):
        """ Finds the max qvalue from the qtable for all actions

        Returns:
            float -- the qvalue
        """
        max_val = max(self.__q_values.values())
        keys = []

        for k, v in self.__q_values.items():
            if v == max_val:
                keys.append(k)

        action = random.choice(keys)

        return self.get_q_value(action), action
