import unittest
import hashlib
from qlearning.qtableentry import QTableEntry


class TestQTableEntry(unittest.TestCase):
    def test_given_new_qtable_entry_get_hash(self):

        state = [1, 2, 3]
        state_str = str(state).encode('utf-8')

        hash_object = hashlib.sha256(state_str)

        qtable_entry = QTableEntry(state)

        self.assertEqual(hash_object.hexdigest(), qtable_entry.get_hash())

    def test_given_new_qtable_entry_check_next_vector_is_false(self):
        state = [1, 2, 3]
        qtable_entry = QTableEntry(state)

        qtable_entry = QTableEntry(state)

        self.assertEqual(False, qtable_entry.has_next_vector())

    def test_given_new_qtable_entry_check_next_vector_raises_exception(self):
        state = [1, 2, 3]
        qtable_entry = QTableEntry(state)

        qtable_entry = QTableEntry(state)

        with self.assertRaises(ValueError):
            qtable_entry.next_vector()

    def test_given_identical_qtable_entry_add_new_state_vector_raises_exception(self):
        state1 = [1, 2, 3]
        qtable_entry1 = QTableEntry(state1)

        state2 = [1, 2, 3]
        qtable_entry2 = QTableEntry(state2)

        with self.assertRaises(ValueError):
            qtable_entry1.add_next(qtable_entry2)

    def test_given_new_qtable_entry_update_q_value_and_check_assert_new_max(self):
        state = [1, 2, 3]
        qtable_entry = QTableEntry(state)

        qtable_entry.set_q_value('UP', 1)

        self.assertEqual(1, qtable_entry.get_q_value_max())


if __name__ == '__main__':
    unittest.main()
