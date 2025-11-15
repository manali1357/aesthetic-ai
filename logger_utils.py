"""
Custom logging configuration with timed rotation and cleanup.

This module provides a custom logging setup with daily log rotation
and automatic cleanup of old log files.
"""

import glob
import logging
import os
import re
from logging.handlers import TimedRotatingFileHandler


# Ensure logs directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


class CustomTimedRotatingFileHandler(TimedRotatingFileHandler):
    """Custom file handler with automatic log cleanup."""
    
    def __init__(self, filename: str, when: str = "midnight", 
                 interval: int = 1, backup_count: int = 7, 
                 encoding: str = "utf-8", keep_days: int = 7):
        """
        Initialize the custom timed rotating file handler.
        
        Args:
            filename: Base filename for the log file
            when: When to rotate ('midnight', 'H', 'M', etc.)
            interval: Interval between rotations
            backup_count: Number of backup files to keep
            encoding: File encoding
            keep_days: Number of days to keep old log files
        """
        super().__init__(filename, when=when, interval=interval, 
                        backupCount=backup_count, encoding=encoding)
        self.keep_days = keep_days
        self.log_dir = os.path.dirname(filename)
        self.base_filename = os.path.basename(filename).replace('.log', '')

    def do_rollover(self) -> None:
        """Perform rollover and cleanup old logs."""
        super().doRollover()
        self._cleanup_old_logs()

    def _cleanup_old_logs(self) -> None:
        """Remove old log files beyond the retention period."""
        # Match files like applog-2025-04-07.log
        log_files = glob.glob(os.path.join(self.log_dir, f"{self.base_filename}-*.log"))
        
        # Extract date part for sorting
        def extract_date(filename: str) -> str:
            """Extract date from filename for sorting."""
            match = re.search(rf"{self.base_filename}-(\d{{4}}-\d{{2}}-\d{{2}})\.log$", filename)
            return match.group(1) if match else ""

        # Sort files by date descending
        log_files.sort(key=extract_date, reverse=True)

        # Remove logs beyond the retention period
        for old_log in log_files[self.keep_days:]:
            try:
                os.remove(old_log)
            except OSError as e:
                logging.warning(f"Failed to delete old log file {old_log}: {e}")


def setup_logger(name: str = "appname", level: int = logging.INFO) -> logging.Logger:
    """
    Initialize and configure the logger with daily rotation.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Define log file path
    log_filename = os.path.join(LOG_DIR, "applog.log")

    # Set up a timed rotating file handler (rotates daily at midnight)
    log_handler = CustomTimedRotatingFileHandler(
        log_filename, 
        when="midnight", 
        interval=1, 
        backup_count=7, 
        encoding="utf-8",
        keep_days=7
    )

    # Customize log file naming
    log_handler.suffix = "%Y-%m-%d"
    log_handler.namer = lambda name: name.replace(".log.", "-") + ".log"

    # Define log format
    log_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    log_handler.setFormatter(log_formatter)

    # Get logger and configure it
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(log_handler)
    logger.info("Logger initialized with daily log rotation.")
    
    return logger


# Initialize the logger
logger = setup_logger()