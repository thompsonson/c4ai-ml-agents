"""Reasoning approaches for ML Agents experiments.

This module provides various reasoning approaches for enhancing model responses,
including Chain-of-Thought, Program-of-Thought, and other advanced reasoning
methodologies.
"""

import importlib
from pathlib import Path
from typing import Dict, List, Type

from ml_agents.config import ExperimentConfig
from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)

# Registry of available reasoning approaches
REASONING_REGISTRY: Dict[str, Type[BaseReasoning]] = {}


def register_reasoning_approach(name: str, approach_class: Type[BaseReasoning]) -> None:
    """Register a reasoning approach in the global registry.

    Args:
        name: Name of the reasoning approach (e.g., "Chain-of-Thought")
        approach_class: The reasoning approach class
    """
    REASONING_REGISTRY[name] = approach_class
    logger.debug(f"Registered reasoning approach: {name}")


def get_available_approaches() -> List[str]:
    """Get list of all available reasoning approach names.

    Returns:
        List of registered reasoning approach names
    """
    return list(REASONING_REGISTRY.keys())


def create_reasoning_approach(name: str, config: ExperimentConfig) -> BaseReasoning:
    """Create a reasoning approach instance by name.

    Args:
        name: Name of the reasoning approach to create
        config: Experiment configuration

    Returns:
        Instance of the requested reasoning approach

    Raises:
        KeyError: If the reasoning approach is not registered
        ValueError: If the approach name is invalid
    """
    if not name:
        raise ValueError("Reasoning approach name cannot be empty")

    if name not in REASONING_REGISTRY:
        available = ", ".join(get_available_approaches())
        raise KeyError(
            f"Unknown reasoning approach: '{name}'. "
            f"Available approaches: {available}"
        )

    approach_class = REASONING_REGISTRY[name]
    logger.info(f"Creating reasoning approach: {name}")
    return approach_class(config)


def _auto_discover_approaches() -> None:
    """Automatically discover and register reasoning approaches.

    This function scans the reasoning module directory for approach
    implementations and automatically registers them.
    """
    reasoning_dir = Path(__file__).parent

    # Look for Python files in the reasoning directory
    for py_file in reasoning_dir.glob("*.py"):
        if py_file.name.startswith("_") or py_file.name == "base.py":
            continue

        module_name = py_file.stem
        try:
            # Dynamically import the module
            module = importlib.import_module(f"ml_agents.reasoning.{module_name}")

            # Look for classes that inherit from BaseReasoning
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseReasoning)
                    and attr is not BaseReasoning
                ):
                    # Register the approach using its class name
                    approach_name = attr_name.replace("Reasoning", "")
                    register_reasoning_approach(approach_name, attr)

        except ImportError as e:
            logger.warning(f"Could not import reasoning module {module_name}: {e}")
        except Exception as e:
            logger.error(f"Error processing reasoning module {module_name}: {e}")


# Auto-discover approaches when the module is imported
_auto_discover_approaches()
