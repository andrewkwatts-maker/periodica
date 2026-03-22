"""
Centralized logging setup for Periodics application.
"""
import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger with consistent formatting for the Periodics application.

    Args:
        name: Module name (e.g., 'data.element_loader', 'main')

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"periodica.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
