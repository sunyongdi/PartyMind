#!/usr/bin/env python
import logging
import datetime

def my_log():
    """配置log打印信息
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 创建Handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    # 文件Handler
    today = datetime.date.today().strftime('%Y-%m-%d')
    filename = f'run_{today}.log'
    fileHandler = logging.FileHandler(filename, mode='a+', encoding='UTF-8')
    fileHandler.setLevel(logging.NOTSET)
    # Formatter
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    # 添加到Logger中
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger