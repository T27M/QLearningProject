import unittest
from core.configbase import ConfigBase
from core.config import Config


class ExampleDerivedClass(ConfigBase):
    def __init__(self, config):
        super().__init__(config=config)

    def get_config(self):
        return self._config


class TestConfigBase(unittest.TestCase):
    def test_given_derived_class_get_correct_config(self):

        config_path = '/test/core_test/configbase_test.json'

        config = Config(config_path)

        derived_class = ExampleDerivedClass(config)

        derived_config_value = derived_class.get_config()['test_value']

        self.assertEqual(1.0, derived_config_value)
