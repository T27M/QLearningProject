import unittest
from core.config import Config
from core.enviroment import Environment


class EnvironmentTest(unittest.TestCase):
    def test_given_no_episodes_raise_exception_on_get_score_over_time(self):

        config_path = '/test/core_test/environment_test.json'
        config = Config(config_path)

        env = Environment(config, None, None)

        with self.assertRaises(ZeroDivisionError):
            env.get_score_over_time()


if __name__ == '__main__':
    unittest.main()
