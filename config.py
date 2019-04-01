import configparser

class Config():

    # variables statiques pour ne charger la config qu'une seule fois
    _config = None
    _section = None

    def __init__(self, section=None):
        if Config._config == None:
            Config._config = configparser.ConfigParser()
            Config._config.read("config.ini")
            if section == None:
                Config._section = Config._config["DEFAULT"]["section"]
            else:
                Config._section = section
            print("Loading config", Config._section)

    def set_config(self, key, val):
        """
            Modification locale, sans modification du fichier de config
        """
        Config._config[key]=str(val)

    def get_config(self, key):
        return Config._config[Config._section][key]

    def get_config_eval(self, key):
        return eval(self.get_config(key))

