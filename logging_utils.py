"""
Logging utilities for the deepfake detection system.
"""

import logging
import logging.handlers
from pathlib import Path
from config import LOGGING_CONFIG


def setup_logger(name: str = "deepfake_detection", level: str = None) -> logging.Logger:
    """
    Set up and configure logger for the application.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    if level is None:
        level = LOGGING_CONFIG.get("level", "INFO")
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # File handler
    log_file = LOGGING_CONFIG.get("log_file")
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5
        )
        file_handler.setLevel(getattr(logging, level.upper()))
    
    # Formatter
    formatter_str = LOGGING_CONFIG.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter(formatter_str)
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
