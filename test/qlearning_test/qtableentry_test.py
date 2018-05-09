import unittest
import hashlib
from qlearning.qtableentry import QTableEntry


class TestQTableEntry(unittest.TestCase):
    def test_given_new_qtable_entry_get_state_str(self):

        state = [1, 2, 3]
        state_str = str(state)

        # hash_object = hashlib.sha256(state_str)

        qtable_entry = QTableEntry(state, [])

        self.assertEqual(state_str, qtable_entry.get_state_str())

    def test_given_new_qtable_entry_update_q_value_and_check_assert_new_max(self):
        state = [1, 2, 3]
        qtable_entry = QTableEntry(state, ['UP', 'DOWN'])

        qtable_entry.set_q_value('UP', 1)

        max_value, action = qtable_entry.get_q_value_max()

        self.assertEqual(1, max_value)
        self.assertEqual(action, 'UP')


if __name__ == '__main__':
    unittest.main()
