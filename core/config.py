import json
from pprint import pprint
import os
import time


class Config(object):
    def __init__(self, config_path='config.json'):

        self.__config_path = os.getcwd() + "/" + config_path

        with open(self.__config_path) as config_file:
            self.config = json.load(config_file)

        self.gen_dir()

    def get_config(self, name):

        if name not in self.config:
            raise ValueError("Could not find config object with key: " + name)

        return self.config[name]

    def gen_dir(self):
        self._timestr = time.strftime("%Y%m%d-%H%M%S")

        self._data_path = './data/qtable/' + self._timestr + '/'

        if not os.path.exists(self._data_path):
            os.makedirs(self._data_path)

    def get_data_dir(self):
        return self._data_path
