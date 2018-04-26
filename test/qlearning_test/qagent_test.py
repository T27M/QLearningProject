import unittest
from core.config import Config
from qlearning.qagent import QAgent


class TestQAgent(unittest.TestCase):
    def test_given_new_q_agent_correctly_update_q_values(self):
        config_path = '/test/qlearning_test/qagent_test.json'

        config = Config(config_path)
        qagent = QAgent(config)

        s = [1, 1]
        s1 = [1, 2]

        """
        Test ran with the following config values:
            discount_factor: 0.95,
            learning_rate: 0.8,
            random_action: 0.1
            
            current_q_value: 0
            reward: 1
            max_q_value_s1: 0

        Q-Learning Algorithm:
            ((1 - 0.8) * 0 + 0.8 * ( 1  + 0.95 * 0) = 0.8
        """

        qagent.update_q_table(s, s1, 'UP', 1)

        self.assertEqual(0.8, qagent.get_q_value(s, 'UP'))


if __name__ == '__main__':
    unittest.main()
