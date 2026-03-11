import logging
import logging.config
import os
from datetime import datetime

def setup_logger(root_dir, log_file_name):
    # print("start setup_logger\n")
    """
    全局的logger，所有的log都通过这个logger输出
    
    Args:
        root_dir (str)      : 整个项目的根目录
        log_file_name (str) : 当前任务的log文件名
    """

    log_dir = root_dir + "/logs/"
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file_path = log_dir + log_file_name + "_" + datetime.now().strftime("%d_%H%M%S") + ".log"


    LOGGING_CONFIG = {
        

        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "standard": {
                "format": "%(levelname)s--%(module)s.%(funcName)s:   %(message)s"
            },
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "filename": log_file_path,
                "encoding": "utf-8",
            },
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
            },
        },
        "root": {
            "handlers": ["file"],
            "level": "DEBUG",
        },
    }


    logging.config.dictConfig(LOGGING_CONFIG)
    # 在日志文件第一行写入初始化信息
    logger = logging.getLogger()
    logger.info("日志系统初始化完成")
    # print("finish setup_logger\n")

    