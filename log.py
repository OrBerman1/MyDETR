import logging
import sys


# Set up logging using basicConfig
def setup_logger(log_file="training_log.txt"):
    # Set up the logger with basicConfig
    logging.basicConfig(
        level=logging.DEBUG,  # Set the log level
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Log to a file
            logging.StreamHandler(sys.stdout)  # Log to console
        ]
    )


# Redirect print statements to the logger
class PrintLogger:
    def __init__(self):
        self.logger = logging.getLogger()  # Use the root logger

    def write(self, message):
        # Avoid logging empty lines or unwanted messages
        if message.strip() != "":
            self.logger.info(message.strip())

    def flush(self):
        pass  # We don't need to do anything here for now


# Set up logging
def creat_logger(log_file):
    setup_logger(log_file)
    sys.stdout = PrintLogger()
