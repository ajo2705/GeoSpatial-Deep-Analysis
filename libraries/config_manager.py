import os
import yaml

from constants import Base, ConfigParams


class ConfigError(Exception):
    def __init__(self, msg):
        self.msg = msg
        super().__init__(self.msg)


class ConfigManager:
    def __init__(self, config_file=ConfigParams.CONFIG_FILE):
        self.config_file = config_file
        self.config_dict = self.__load_config()

    def __load_config(self):
        """
        Reads yaml file and returns content as configurations dict
        :return: dict
        """
        res_path = Base.BASE_RESOURCE_PATH
        with open(os.path.join(res_path + self.config_file)) as f:
            configurations = yaml.safe_load(f)

        return configurations

    def get_config_parameter(self, parameter_name, default_value=None):
        content = self.config_dict

        for val in parameter_name.split("/"):
            if val in content:
                content = content[val]
            else:
                if default_value:
                    return default_value

                msg = f"{parameter_name} - position {val} missing in config file {self.config_file}"
                raise ConfigError(msg)

        return content
