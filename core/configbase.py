import time


class ConfigBase(object):
    def __init__(self, config):
        self._debug = config.get_config('debug')
        self._log_output = config.get_config('log_output')
        self._timestr = time.strftime("%Y%m%d-%H%M%S")
        self._log_path = config.get_config('log_path') + \
            type(self).__name__ + self._timestr + ".txt"

        self._config = config.get_config(type(self).__name__)
