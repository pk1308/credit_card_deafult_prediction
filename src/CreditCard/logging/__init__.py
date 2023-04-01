import os
from datetime import datetime

from loguru import logger

LOG_DIR = "logs"

get_current_time_stamp = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
get_log_file_name = f"log_{get_current_time_stamp}.log"


LOG_FILE_NAME = get_log_file_name

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)
log_format = "{time:YYYY-MM-DD at HH:mm:ss} | {level} |{function} | {message}"
logger.add(LOG_FILE_PATH, format=log_format, level="DEBUG")