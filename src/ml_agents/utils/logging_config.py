"""Logging configuration for ML Agents."""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Human-readable formatter with colors for console output."""

    # Color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def __init__(self, use_colors: bool = True):
        """Initialize formatter.

        Args:
            use_colors: Whether to use colors in output
        """
        super().__init__()
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human reading."""
        # Create timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        # Format level with colors if enabled
        level = record.levelname
        if self.use_colors:
            color = self.COLORS.get(level, "")
            reset = self.COLORS["RESET"]
            level = f"{color}{level}{reset}"

        # Format location info
        location = f"{record.module}:{record.funcName}:{record.lineno}"

        # Format message
        message = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            exception_str = self.formatException(record.exc_info)
            message = f"{message}\n{exception_str}"

        return f"{timestamp} | {level:>8} | {location:>20} | {message}"


def setup_logging(
    level: Optional[str] = None,
    format_type: Optional[str] = None,
    log_to_file: Optional[bool] = None,
    log_dir: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('human' or 'json')
        log_to_file: Whether to log to files
        log_dir: Directory for log files
        max_file_size: Maximum size of log files in bytes
        backup_count: Number of backup files to keep

    Returns:
        Configured logger
    """
    # Get configuration from environment if not provided
    level = level or os.getenv("LOG_LEVEL", "ERROR").upper()
    format_type = format_type or os.getenv("LOG_FORMAT", "human").lower()
    log_to_file = (
        log_to_file
        if log_to_file is not None
        else os.getenv("LOG_TO_FILE", "false").lower() == "true"
    )
    log_dir = log_dir or os.getenv("LOG_DIR", "logs")

    # Validate level
    numeric_level = getattr(logging, level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # Create log directory if needed
    if log_to_file:
        Path(log_dir or "logs").mkdir(parents=True, exist_ok=True)

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Set up console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)

    if format_type == "json":
        console_formatter: logging.Formatter = JSONFormatter()
    else:
        console_formatter = HumanFormatter(use_colors=True)

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Set up file handler if enabled
    if log_to_file:
        log_file = Path(log_dir or "logs") / "ml_agents.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)

        # Always use JSON format for file logging
        file_formatter = JSONFormatter()
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Configure third-party loggers to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_experiment_start(
    config: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> None:
    """Log experiment start with configuration.

    Args:
        config: Experiment configuration
        logger: Logger to use (defaults to root logger)
    """
    if logger is None:
        logger = get_logger(__name__)

    logger.info(
        "Starting ML Agents experiment",
        extra={"extra_fields": {"event": "experiment_start", "config": config}},
    )


def log_experiment_end(
    results: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> None:
    """Log experiment end with results summary.

    Args:
        results: Experiment results summary
        logger: Logger to use (defaults to root logger)
    """
    if logger is None:
        logger = get_logger(__name__)

    logger.info(
        "Completed ML Agents experiment",
        extra={"extra_fields": {"event": "experiment_end", "results": results}},
    )


def log_api_call(
    provider: str,
    model: str,
    duration: float,
    tokens: int,
    cost: float,
    success: bool = True,
    error: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log API call details.

    Args:
        provider: API provider name
        model: Model name
        duration: Call duration in seconds
        tokens: Number of tokens used
        cost: Cost in USD
        success: Whether the call was successful
        error: Error message if call failed
        logger: Logger to use (defaults to root logger)
    """
    if logger is None:
        logger = get_logger(__name__)

    level = logging.INFO if success else logging.ERROR
    message = f"API call to {provider}/{model}"

    extra_fields = {
        "event": "api_call",
        "provider": provider,
        "model": model,
        "duration": duration,
        "tokens": tokens,
        "cost": cost,
        "success": success,
    }

    if error:
        extra_fields["error"] = error

    logger.log(level, message, extra={"extra_fields": extra_fields})


def log_reasoning_step(
    step_name: str,
    prompt: str,
    response: str,
    duration: float,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log reasoning step details.

    Args:
        step_name: Name of the reasoning step
        prompt: Input prompt
        response: Model response
        duration: Step duration in seconds
        logger: Logger to use (defaults to root logger)
    """
    if logger is None:
        logger = get_logger(__name__)

    logger.debug(
        f"Reasoning step: {step_name}",
        extra={
            "extra_fields": {
                "event": "reasoning_step",
                "step_name": step_name,
                "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                "response": response[:500] + "..." if len(response) > 500 else response,
                "duration": duration,
            }
        },
    )


# Initialize logging on import if not already configured
if not logging.getLogger().handlers:
    setup_logging()
