import os
import logging
import sys
import time
from configuration.config import config

class Logger:
    def __init__(self):
        self.logger = logging.getLogger('NAS')
        self.logger.setLevel(getattr(logging,config.LOG_LEVEL))
        self.file_handler = None
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(self.console_handler)
        
    def setup_file_logging(self):
        if self.file_handler is not None:
            return
        
        if not os.path.exists(config.LOG_DIR):
            os.makedirs(config.LOG_DIR)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.file_handler = logging.FileHandler(
            os.path.join(config.LOG_DIR,f'nas_{timestamp}.log'),
            encoding="utf-8"
        )
        self.file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        self.logger.addHandler(self.file_handler)
        
    def info(self, msg: str):
        self.logger.info(msg)
        
    def debug(self, msg: str):
        self.logger.debug(msg)
        
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
        
logger = Logger()