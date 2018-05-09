import unittest
from core.config import Config
from qlearning.qagent import QAgent


class TestQAgent(unittest.TestCase):
    def test_given_new_q_agent_correctly_update_q_values(self):
        config_path = '/test/qlearning_test/qagent_test.json'
        config = Config(config_path)

        qagent = QAgent(config)

        state = [1, 1]
        next_state = [1, 2]

        """
        Test ran with the following config values:
            discount_factor: 0.95,
            learning_rate: 0.8,
            random_action: 0.0
            
            current_q_value: 0
            reward: 1
            max_q_value_s1: 0

        Q-Learning Algorithm:
            q(s) + lr * (reward + df * maxq(s1) - q(s))
            (0 + 0.8 * (1  + 0.95 * 0 - 0) = 0.8
        """

        qagent.update_q_table(state, next_state, 'UP', 1)

        self.assertEqual(0.8, qagent.get_q_value(state, 'UP'))

    def test_given_new_q_agent_correctly_update_q_values_multiple(self):
        config_path = '/test/qlearning_test/qagent_test.json'
        config = Config(config_path)

        qagent = QAgent(config)

        state = [1, 1]
        next_state = [1, 2]

        """
        Test ran with the following config values:
            discount_factor: 0.95,
            learning_rate: 0.8,
            random_action: 0.0
            
            current_q_value: 0.8
            reward: 5
            max_q_value_s1: 0

        Q-Learning Algorithm:
            q(s) + lr * (reward + df * maxq(s1) - q(s))
            0.8 + 0.8 * (5  + 0.95 * 0 - 0.8) = 0.8
        """

        qagent.update_q_table(state, next_state, 'UP', 1)
        qagent.update_q_table(state, next_state, 'UP', 5)

        self.assertEqual(4.16, qagent.get_q_value(state, 'UP'))

    def test_given_new_q_agent_no_random_correctly_predict_greedy_action(self):
        config_path = '/test/qlearning_test/qagent_test.json'
        config = Config(config_path)

        state = [1, 1]
        next_state = [1, 2]

        """
        Test ran with the following config values:
            discount_factor: 0.95,
            learning_rate: 0.8,
            random_action: 0.0
        """
        qagent = QAgent(config)
        qagent.update_q_table(state, next_state, 'UP', 1)

        action = qagent.predict(state)

        self.assertEqual('UP', action)


if __name__ == '__main__':
    unittest.main()
