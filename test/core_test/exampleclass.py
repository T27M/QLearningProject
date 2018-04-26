from core.configbase import ConfigBase


class ExampleDerivedClass(ConfigBase):
    def __init__(self, config):
        super().__init__(config=config)

    def get_config(self):
        return self._config
