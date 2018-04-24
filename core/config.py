import json
from pprint import pprint
import os


class Config(object):
    def __init__(self, config_path='config.json'):

        self.__config_path = os.getcwd() + "/" + config_path

        with open(self.__config_path) as config_file:
            self.config = json.load(config_file)

    def get_config(self, name):

        if name not in self.config:
            raise ValueError("Could not find config object with key: " + name)

        return self.config[name]
