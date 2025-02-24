import logging
import logging.config

msglogger = logging.getLogger("debug")

def msglogger_init(log_path, log_name, log_level="DEBUG"):
    ''''''
    log_level = log_level.upper()

    LOG_PATH_DEBUG = "%s/%s_debug.log" % (log_path, log_name)
    LOG_FILE_MAX_BYTES = 1 * 128 * 1024 * 1024
    LOG_FILE_BACKUP_COUNT = 30

    log_conf = {
        "version": 1,
        "formatters": {
            "format1": {
                "format": '%(message)s',
            },
        },

        "handlers": {

            "handler1": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "format1",
                "maxBytes": LOG_FILE_MAX_BYTES,
                "backupCount": LOG_FILE_BACKUP_COUNT,
                "filename": LOG_PATH_DEBUG
            },

        },

        "loggers": {

            "debug": {
                "handlers": ["handler1"],
                "level": log_level
            },
        },
    }
    logging.config.dictConfig(log_conf)