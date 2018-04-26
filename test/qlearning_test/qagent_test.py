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

        qagent.update_q_table(s, s1, 'UP', 1)

        print(qagent.get_q_values(s))


if __name__ == '__main__':
    unittest.main()
