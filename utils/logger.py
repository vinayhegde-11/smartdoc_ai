import logging
from logging.handlers import RotatingFileHandler
import os

def get_logger(name:str):
    # create log directory in the project root
    os.makedirs('logs', exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # create handlers only if they don't already exist
    if not logger.handlers:
        file_handler = RotatingFileHandler(
            filename='logs/pipeline.log',
            maxBytes=5*1024*1024,
            backupCount=3,
            encoding='utf-8')
    
        # set formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger