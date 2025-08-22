# Phase 10: Python Package Distribution

## Strategic Context

**Purpose**: Package ML Agents as a distributable Python package for easy installation via pip, uv, and uvx.

**Integration Point**: Enables researchers to install with `pip install ml-agents-reasoning` or `uvx ml-agents-reasoning` without cloning the repository.

**Timeline**: 4-5 hours total implementation

## Phase 10 Strategic Decisions

### **Primary Implementation: PyPI Distribution**

**Decision**: Package as `ml-agents-reasoning` on PyPI with full CLI support via entry points.

**Rationale**:

- Easy installation for researchers without Git/development setup
- Version management and dependency resolution
- Global CLI access via `ml-agents` command
- Supports both traditional pip and modern uv/uvx workflows

### **Package Structure**

**Enhanced pyproject.toml**:

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ml-agents-reasoning"
version = "0.1.0"
description = "ML Agents Reasoning Research Platform"
readme = "README.md"
license = {text = "Creative Commons Attribution 4.0 International"}
authors = [
    {name = "Cohere Labs", email = "labs@cohere.com"}
]
keywords = ["reasoning", "llm", "benchmark", "research", "ai"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

requires-python = ">=3.8"
dependencies = [
    "transformers>=4.35.0",
    "accelerate>=0.24.0",
    "openai>=1.3.0",
    "cohere>=4.37.0",
    "anthropic>=0.8.0",
    "pandas>=2.1.0",
    "datasets>=2.14.0",
    "python-dotenv>=1.0.0",
    "tqdm>=4.66.0",
    "huggingface_hub>=0.19.0",
    "pyyaml>=6.0.1",
    "torch>=2.1.0",
    "typer>=0.9.0",
    "rich>=13.0.0",
    "instructor>=1.0.0",
    "pydantic>=2.0.0",
    "sqlalchemy>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "black>=23.0.0",
    "mypy>=1.5.0",
    "pytest>=7.4.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.11.0",
    "pytest-asyncio>=0.21.0",
    "flake8>=6.0.0",
    "isort>=5.12.0",
    "pre-commit>=3.4.0",
    "build>=0.10.0",
    "twine>=4.0.0",
]

[project.urls]
Homepage = "https://github.com/thompsonson/c4ai-ml-agents"
Documentation = "https://github.com/thompsonson/c4ai-ml-agents#readme"
Repository = "https://github.com/thompsonson/c4ai-ml-agents"
Issues = "https://github.com/thompsonson/c4ai-ml-agents/issues"

[project.scripts]
ml-agents = "ml_agents.cli.main:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
ml_agents = ["reasoning/prompts/*.txt", "configs/*.yaml"]
```

## Technical Implementation

### **Phase 10.1: Package Structure (1.5 hours)**

**Directory Restructuring**:

```
ml-agents-reasoning/
├── src/
│   └── ml_agents/           # Package name with underscores
│       ├── __init__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py      # Entry point
│       ├── core/
│       ├── reasoning/
│       └── utils/
├── tests/
├── docs/
├── pyproject.toml
├── README.md
├── LICENSE
└── MANIFEST.in
```

**Entry Point Configuration**:

```python
# src/ml_agents/cli/main.py
import typer
from ml_agents.core.experiment_runner import ExperimentRunner
from ml_agents.core.experiment_config import ExperimentConfig

app = typer.Typer(name="ml-agents", help="ML Agents Reasoning Research Platform")

# Import all CLI commands
from ml_agents.cli.run_commands import run_app
from ml_agents.cli.preprocess_commands import preprocess_app

app.add_typer(run_app, name="run")
app.add_typer(preprocess_app, name="preprocess")

if __name__ == "__main__":
    app()
```

### **Phase 10.2: Build System (1 hour)**

**Build Configuration**:

```python
# MANIFEST.in
include README.md
include LICENSE
recursive-include src/ml_agents/reasoning/prompts *.txt
recursive-include src/ml_agents/configs *.yaml
recursive-exclude * __pycache__
recursive-exclude * *.py[co]
```

**Version Management**:

```python
# src/ml_agents/__init__.py
__version__ = "0.1.0"
__author__ = "Cohere Labs"
__email__ = "labs@cohere.com"
```

### **Phase 10.3: Distribution (1.5 hours)**

**Build and Upload Scripts**:

```bash
#!/bin/bash
# scripts/build_and_upload.sh

echo "Building ML Agents package..."

# Clean previous builds
rm -rf dist/ build/ src/*.egg-info/

# Build package
python -m build

# Check package
python -m twine check dist/*

# Upload to PyPI (requires API token)
python -m twine upload dist/*

echo "Package uploaded successfully!"
```

**Installation Testing**:

```bash
# Test different installation methods
pip install ml-agents-reasoning
uv add ml-agents-reasoning
uvx ml-agents-reasoning --help
```

## Installation Methods

### **Standard pip Installation**

```bash
# Install globally
pip install ml-agents-reasoning

# Install with development dependencies
pip install ml-agents-reasoning[dev]

# Verify installation
ml-agents --version
ml-agents --help
```

### **Modern uv Installation**

```bash
# Add to project dependencies
uv add ml-agents-reasoning

# Install globally with uv
uv tool install ml-agents-reasoning

# Run without installing (uvx)
uvx ml-agents-reasoning run --approach CoT --samples 10
```

### **Development Installation**

```bash
# Clone and install in development mode
git clone https://github.com/thompsonson/c4ai-ml-agents
cd c4ai-ml-agents
pip install -e .[dev]

# Or with uv
uv sync --all-extras
```

## Documentation Updates

### **Updated README.md**

```markdown
# ML Agents Reasoning Research Platform

## Installation

### Quick Start
```bash
pip install ml-agents-reasoning
ml-agents --help
```

### Modern Python (uv/uvx)

```bash
# Install with uv
uv tool install ml-agents-reasoning

# Run without installing
uvx ml-agents-reasoning run --approach CoT
```

### From Source

```bash
git clone https://github.com/thompsonson/c4ai-ml-agents
cd c4ai-ml-agents
pip install -e .
```

## Usage

```bash
# Run single experiment
ml-agents run --approach ChainOfThought --samples 50

# Compare approaches
ml-agents compare --approaches "CoT,Reflection" --samples 100

# Preprocess datasets
ml-agents preprocess inspect MilaWang/SpatialEval
```

```

### **PyPI Project Description**:
```markdown
# ML Agents Reasoning Research Platform

A comprehensive framework for evaluating reasoning approaches across language models.

## Features
- 8+ reasoning approaches (Chain-of-Thought, Reflection, Tree-of-Thought, etc.)
- Multi-provider support (OpenAI, Anthropic, Cohere, OpenRouter)
- Automated dataset preprocessing for 77+ benchmarks
- Structured output parsing with fallback mechanisms
- Database persistence and analysis tools
- Rich CLI with progress tracking and cost monitoring

## Quick Example
```python
from ml_agents import ExperimentRunner, ExperimentConfig

config = ExperimentConfig(
    dataset_name="c4ai-ml-agents/SpatialEval",
    reasoning_approaches=["ChainOfThought", "Reflection"],
    sample_count=50
)

runner = ExperimentRunner(config)
results = runner.run_comparison()
```

```

## Release Strategy

### **Phase 10.4: Initial Release (1 hour)**

**Version 0.1.0 Features**:
- Core reasoning approaches (8 implemented)
- CLI interface with all commands
- Dataset preprocessing pipeline
- Database persistence
- Output parsing with Instructor

**Release Checklist**:
- [ ] Package builds successfully
- [ ] All tests pass
- [ ] CLI commands work post-installation
- [ ] Documentation complete
- [ ] PyPI upload successful
- [ ] Installation tested on clean environments

**Future Versions**:
- 0.2.0: Additional reasoning approaches (Graph-of-Thought, ReWOO, etc.)
- 0.3.0: Web dashboard and advanced analysis
- 1.0.0: Production-ready with comprehensive benchmarks

## Success Criteria

**Installation Success**:
- Package installs successfully via pip, uv, and uvx
- CLI commands accessible globally after installation
- All dependencies resolved correctly
- Works on Python 3.8-3.11

**Distribution Success**:
- Listed on PyPI as `ml-agents-reasoning`
- Download statistics tracking enabled
- Package metadata complete and accurate
- Installation instructions clear and tested

**User Experience**:
- Zero-config installation for basic usage
- Clear error messages for missing dependencies
- Comprehensive help documentation
- Quick start examples work out of the box

**Documentation Complete**:
- README updated with installation methods
- PyPI description comprehensive
- CLI help text accurate
- Examples tested and working

This phase transforms the project from a development repository into a professional, distributable research tool accessible to the broader AI research community.
