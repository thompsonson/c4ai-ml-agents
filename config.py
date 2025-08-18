"""Configuration module for loading environment variables.

DEPRECATED: This module is deprecated. Use src.config instead.
This file is maintained for backward compatibility with the Jupyter notebook.
"""

import sys
import warnings
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import from new location
from src.config import (
    API_KEYS,
    SUPPORTED_MODELS,
    SUPPORTED_REASONING,
    ExperimentConfig,
    get_api_key,
    get_default_config,
    validate_api_keys,
    validate_environment,
)

# Issue deprecation warning
warnings.warn(
    "config.py is deprecated. Use 'from src.config import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Maintain backward compatibility by exposing all original functions
__all__ = [
    "API_KEYS",
    "get_api_key",
    "validate_api_keys",
    "ExperimentConfig",
    "SUPPORTED_MODELS",
    "SUPPORTED_REASONING",
    "get_default_config",
    "validate_environment",
]
