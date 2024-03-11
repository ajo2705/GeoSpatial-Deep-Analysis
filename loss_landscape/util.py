from enum import Enum
from libraries.config_manager import ConfigError


def load_none_on_error(func):
    def inner(*args, **kwargs):
        try:
            print(func)
            return func(*args, **kwargs)

        except ConfigError as ex:
            print(ex)
            return None

    return inner


class Models(Enum):
    Model1 = 0
    Model2 = 1
    Model3 = 2
