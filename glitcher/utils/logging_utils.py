#!/usr/bin/env python3
"""
Logging utilities for glitch token classification

This module provides standardized logging setup and utilities used throughout
the classification system, including tqdm-compatible logging.
"""

import logging
from pathlib import Path
from typing import Optional
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that works with tqdm progress bars"""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logger(
    name: str = "GlitchClassifier",
    log_file: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    enable_file_logging: bool = True
) -> logging.Logger:
    """
    Set up a logger with both console and file output

    Args:
        name: Logger name
        log_file: Path to log file (default: glitch_classifier.log)
        console_level: Logging level for console output
        file_level: Logging level for file output
        enable_file_logging: Whether to enable file logging

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Set up console handler with tqdm compatibility
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Set up file handler with timestamps if enabled
    if enable_file_logging:
        if log_file is None:
            log_file = "glitch_classifier.log"

        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False

    return logger


def get_logger(name: str = "GlitchClassifier") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        return setup_logger(name)

    return logger


def set_log_level(logger: logging.Logger, level: int):
    """
    Set the logging level for both console and file handlers

    Args:
        logger: Logger instance
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        if isinstance(handler, TqdmLoggingHandler):
            handler.setLevel(level)


def enable_debug_logging(logger: logging.Logger):
    """Enable debug-level logging for detailed output"""
    set_log_level(logger, logging.DEBUG)


def disable_debug_logging(logger: logging.Logger):
    """Disable debug-level logging, keep info and above"""
    set_log_level(logger, logging.INFO)


def log_test_start(logger: logging.Logger, token: str, token_id: int, test_name: str):
    """Log the start of a classification test"""
    logger.debug(f"Starting {test_name} for token '{token}' (ID: {token_id})")


def log_test_result(
    logger: logging.Logger,
    token: str,
    test_name: str,
    category: str,
    is_positive: bool,
    indicators: dict = None
):
    """Log the result of a classification test"""
    status = "POSITIVE" if is_positive else "NEGATIVE"
    logger.debug(f"Test {test_name} for token '{token}': {status} ({category})")

    if is_positive and indicators:
        triggered = [name for name, triggered in indicators.items() if triggered]
        if triggered:
            logger.debug(f"  Triggered indicators: {', '.join(triggered)}")


def log_classification_summary(
    logger: logging.Logger,
    token: str,
    token_id: int,
    categories: list
):
    """Log the final classification summary for a token"""
    categories_str = ", ".join(categories) if categories else "None"
    logger.info(f"Token '{token}' (ID: {token_id}) classified as: {categories_str}")


def log_error(logger: logging.Logger, message: str, exception: Exception = None):
    """Log an error with optional exception details"""
    if exception:
        logger.error(f"{message}: {exception}")
        logger.debug("Exception details:", exc_info=True)
    else:
        logger.error(message)


def log_warning(logger: logging.Logger, message: str):
    """Log a warning message"""
    logger.warning(message)


def log_info(logger: logging.Logger, message: str):
    """Log an info message"""
    logger.info(message)


def log_banner(logger: logging.Logger, title: str, width: int = 80):
    """Log a banner with title"""
    logger.info("=" * width)
    logger.info(title.center(width))
    logger.info("=" * width)


def log_section(logger: logging.Logger, title: str, width: int = 80):
    """Log a section header"""
    logger.info(f"\n{title}")
    logger.info("-" * min(len(title), width))


class ProgressLogger:
    """Context manager for logging progress with tqdm integration"""

    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        desc: str = "Processing",
        unit: str = "items"
    ):
        self.logger = logger
        self.total = total
        self.desc = desc
        self.unit = unit
        self.pbar = None
        self.count = 0

    def __enter__(self):
        self.pbar = tqdm(total=self.total, desc=self.desc, unit=self.unit)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            self.pbar.close()

    def update(self, n: int = 1, message: str = None):
        """Update progress bar and optionally log a message"""
        self.count += n
        if self.pbar:
            self.pbar.update(n)

        if message:
            self.logger.debug(f"[{self.count}/{self.total}] {message}")

    def set_description(self, desc: str):
        """Update the progress bar description"""
        if self.pbar:
            self.pbar.set_description(desc)


# Default logger instance for backward compatibility
default_logger = None

def get_default_logger() -> logging.Logger:
    """Get the default logger instance, creating it if needed"""
    global default_logger
    if default_logger is None:
        default_logger = setup_logger()
    return default_logger
