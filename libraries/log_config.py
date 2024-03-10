import logging.config
import os

from constants import Base


def setup_logging():
    log_directory = os.path.join(Base.BASE_PATH, "Logs")
    info_log_filename = os.path.join(log_directory, "info.log")
    debug_filename = os.path.join(log_directory, "debug.log")

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        'handlers': {
            'debug_file_handler': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'standard',
                'filename': debug_filename,
                'mode': 'a',
            },
            'info_file_handler': {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'filename': info_log_filename,
                'mode': 'a',
            },
        },
        'loggers': {
            'cnn_logger': {  # Custom named logger
                'handlers': ['debug_file_handler', 'info_file_handler'],
                'level': 'DEBUG',
                'propagate': False,  # Prevent the log messages from being propagated to the root logger
            },
        }
    })