import time


class ConfigBase(object):
    def __init__(self, config):
        self._debug = config.get_config('debug')

        self._timestr = time.strftime("%Y%m%d-%H%M%S")

        self._config = config.get_config(type(self).__name__)
        self._data_dir = config.get_data_dir()
