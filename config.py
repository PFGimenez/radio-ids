import configparser

class Config():
    def __init__(self):
        self._config = configparser.ConfigParser()
        self._config.read("config.ini")
        self._section = self._config["DEFAULT"]["section"]
        print("Loading config", self._section)

    def get_config(self, key):
        return self._config[self._section][key]

    def get_config_eval(self, key):
        return eval(self.get_config(key))

