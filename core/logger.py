import logging
import os
from typing import Optional

from core.config import load_config

_configured = False


def setup_logging(
    level: Optional[str] = None,
    log_to_file: Optional[bool] = None,
    file_path: Optional[str] = None,
) -> logging.Logger:
    """
    Configure structured logging for the DDx system.
    Sets up the parent logger 'ddx' with handlers and levels.

    Args:
        level (str, optional): Logging level string (e.g. "INFO").
        log_to_file (bool, optional): Whether to write logs to a file.
        file_path (str, optional): Target log file path.

    Returns:
        logging.Logger: The configured root logger instance.
    """
    global _configured

    config = load_config()
    log_cfg = config.get("logging", {})

    log_level_str = level or log_cfg.get("level", "INFO")
    to_file = (
        log_to_file
        if log_to_file is not None
        else log_cfg.get("log_to_file", False)
    )
    f_path = file_path or log_cfg.get("file_path", "results/ddx.log")

    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)

    root_logger = logging.getLogger("ddx")
    root_logger.setLevel(numeric_level)

    if not _configured:
        root_logger.handlers.clear()

        # Timestamped clean log layout
        console_formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(numeric_level)
        root_logger.addHandler(console_handler)

        # File handler
        if to_file:
            log_dir = os.path.dirname(f_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(f_path, encoding="utf-8")
            file_handler.setFormatter(console_formatter)
            file_handler.setLevel(numeric_level)
            root_logger.addHandler(file_handler)

        root_logger.propagate = False
        _configured = True

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a child logger for a specific component under the 'ddx' hierarchy.

    Args:
        name (str): Component identifier.

    Returns:
        logging.Logger: The child logger instance.
    """
    setup_logging()
    return logging.getLogger(f"ddx.{name}")
