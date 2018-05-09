import unittest
from qlearning.qtable import QTable
from qlearning.qtableentry import QTableEntry


class TestQTable(unittest.TestCase):
    def test_given_new_qtable_add_new_q_entry_and_get_qtable(self):
        state = [1, 2, 3]

        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

        qtable = QTable(actions)

        q_table_values = qtable.get_q_table_values(state)

        self.assertIsNotNone(q_table_values)
        self.assertDictEqual(
            {
                'UP': 0.0,
                'LEFT': 0.0,
                'RIGHT': 0.0,
                'DOWN': 0.0
            },
            q_table_values
        )

    def test_given_new_qtable_get_q_value_for_valid_action(self):
        state = [1, 2, 3]
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

        qtable = QTable(actions)

        self.assertEqual(0, qtable.get_q_table_value(state, 'UP'))

    def test_given_new_qtable_get_q_value_for_invalid_action_raises_exception(self):
        state = [1, 2, 3]
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

        qtable = QTable(actions)

        with self.assertRaises(KeyError):
            qtable.get_q_table_value(state, 'NOACTION')

    def test_given_new_qtable_update_q_value_and_assert_correct_value(self):
        state = [1, 2, 3]
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

        qtable = QTable(actions)
        qtable.update_q_table_value(state, 'UP', 1)

        self.assertEqual(1, qtable.get_q_table_value(state, 'UP'))

    def test_give_new_qtable_get_dict_len(self):
        state = [1, 2, 3]
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

        qtable = QTable(actions)
        qtable.update_q_table_value(state, 'UP', 1)

        self.assertEqual(1, qtable.get_q_table_len())


if __name__ == '__main__':
    unittest.main()
