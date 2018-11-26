import configparser

class Config():

    # variables statiques pour ne charger la config qu'une seule fois
    _config = None
    _section = None

    def __init__(self):
        if Config._config == None:
            Config._config = configparser.ConfigParser()
            Config._config.read("config.ini")
            Config._section = Config._config["DEFAULT"]["section"]
            print("Loading config", Config._section)

    def get_config(self, key):
        return Config._config[Config._section][key]

    def get_config_eval(self, key):
        return eval(self.get_config(key))

