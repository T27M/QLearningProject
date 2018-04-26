import hashlib
from qlearning.qtableentry import QTableEntry


class QTable(object):
    def __init__(self):
        self.__q_table = {}

        pass

    def state_exists(self, observation):
        hash_object = hashlib.md5(observation)

        return self.__q_table[hash_object] is not None

    def get_q_value(self, observation):
        hash_object = hashlib.md5(observation)

        q_table_entry = QTableEntry(
            observation
        )

        pass

    def __update_q_values(self, ob, ob1, action, reward):
        # Update Q-Table
        # self.QTable[ob, action] = ((1 - self.learning_rate) * self.QTable[ob, action]) + self.learning_rate * (reward + self.discount_factor * np.max(self.QTable[ob1, :]))

        pass
