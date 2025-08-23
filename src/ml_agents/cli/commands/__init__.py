"""CLI commands package for ML Agents."""

from .db import db_backup, db_init, db_migrate, db_stats
from .eval import (
    list_checkpoints,
    resume_experiment,
    run_comparison_experiment,
    run_single_experiment,
)
from .preprocess import (
    preprocess_batch,
    preprocess_generate_rules,
    preprocess_inspect,
    preprocess_list_unprocessed,
    preprocess_transform,
    preprocess_upload,
)
from .results import (
    analyze_experiment,
    compare_experiments,
    export_experiment,
    list_experiments,
)
from .setup import list_approaches, validate_env, version

__all__ = [
    # Database commands
    "db_backup",
    "db_init",
    "db_migrate",
    "db_stats",
    # Evaluation commands
    "list_checkpoints",
    "resume_experiment",
    "run_comparison_experiment",
    "run_single_experiment",
    # Preprocessing commands
    "preprocess_batch",
    "preprocess_generate_rules",
    "preprocess_inspect",
    "preprocess_list_unprocessed",
    "preprocess_transform",
    "preprocess_upload",
    # Results commands
    "analyze_experiment",
    "compare_experiments",
    "export_experiment",
    "list_experiments",
    # Setup commands
    "list_approaches",
    "validate_env",
    "version",
]
