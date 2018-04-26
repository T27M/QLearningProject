import unittest
from core.config import Config


class ConfigTest(unittest.TestCase):
    def test_given_valid_context_parse_and_return_values(self):

        config_path = '/test/core_test/config_test.json'

        config = Config(config_path)

        config_value_number = config.get_config('Test')['test_config_number']
        config_value_string = config.get_config('Test')['test_config_string']
        config_value_float = config.get_config('Test')['test_config_float']

        self.assertEqual(config_value_number, 1)
        self.assertEqual(config_value_string, "1")
        self.assertEqual(config_value_float, 1.0)

    def test_given_undefined_config_key_raise_exception(self):
        config_path = '/test/core_test/config_test.json'

        config = Config(config_path)

        with self.assertRaises(ValueError):
            config.get_config('Invalid')

    def test_method(self):
        self.assertEquals(1, 1)


if __name__ == '__main__':
    unittest.main()
