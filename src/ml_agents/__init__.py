"""ML Agents Reasoning Research Platform.

A comprehensive framework for evaluating reasoning approaches across language models.
"""

__version__ = "0.1.0"
__author__ = "Cohere Labs"
__email__ = "labs@cohere.com"
__description__ = "ML Agents Reasoning Research Platform"

# Core exports
from ml_agents.config import ExperimentConfig
from ml_agents.core.experiment_runner import ExperimentRunner

__all__ = ["ExperimentConfig", "ExperimentRunner", "__version__"]
