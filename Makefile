# Makefile for ML Agents Project
# Provides convenient commands for development workflow

.PHONY: help install install-dev test test-cov lint format type-check clean check pre-commit setup build docs

# Default target
help:
	@echo "ML Agents Development Commands"
	@echo "============================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup          Complete project setup (install + pre-commit)"
	@echo "  make install        Install package and dependencies"
	@echo "  make install-dev    Install with development dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  make test           Run test suite"
	@echo "  make test-cov       Run tests with coverage report"
	@echo "  make lint           Run all linting tools"
	@echo "  make format         Format code with black and isort"
	@echo "  make type-check     Run mypy type checking"
	@echo "  make check          Run all checks (format, lint, test)"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  make clean          Clean up generated files"
	@echo "  make pre-commit     Install and run pre-commit hooks"
	@echo "  make build          Build distribution packages"
	@echo ""
	@echo "Testing Commands:"
	@echo "  make test-config    Test configuration module"
	@echo "  make test-logging   Test logging module"
	@echo "  make test-backward  Test backward compatibility"

# Variables
VENV_ACTIVATE = source .venv/bin/activate
UV = uv
PYTHON = python
PYTEST = pytest
BLACK = black
ISORT = isort
MYPY = mypy
FLAKE8 = flake8

# Setup and Installation
setup: install-dev pre-commit
	@echo "✅ Project setup complete!"

install:
	@echo "📦 Installing package and dependencies..."
	$(VENV_ACTIVATE) && $(UV) pip install -e .

install-dev:
	@echo "📦 Installing package with development dependencies..."
	$(VENV_ACTIVATE) && $(UV) pip install -e ".[dev]"

# Testing
test:
	@echo "🧪 Running test suite..."
	$(VENV_ACTIVATE) && $(PYTEST) tests/ -v -m "not integration"

test-cov:
	@echo "🧪 Running test suite with coverage..."
	$(VENV_ACTIVATE) && $(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term-missing -m "not integration"

test-integration:
	@echo "🌐 Running integration tests (may call real APIs)..."
	$(VENV_ACTIVATE) && $(PYTEST) tests/ -v -m integration

test-all:
	@echo "🧪 Running all tests including integration..."
	$(VENV_ACTIVATE) && $(PYTEST) tests/ -v

test-config:
	@echo "🧪 Testing configuration module..."
	$(VENV_ACTIVATE) && $(PYTEST) tests/test_config.py -v

test-logging:
	@echo "🧪 Testing logging module..."
	$(VENV_ACTIVATE) && $(PYTEST) tests/utils/test_logging_config.py -v

test-backward:
	@echo "🧪 Testing backward compatibility..."
	$(VENV_ACTIVATE) && $(PYTHON) -c "import config; print('✅ Backward compatibility works!')"
	$(VENV_ACTIVATE) && $(PYTHON) -c "from src.config import ExperimentConfig; print('✅ New imports work!')"

# Code Quality
format:
	@echo "🎨 Formatting code..."
	$(VENV_ACTIVATE) && $(ISORT) src/ tests/
	$(VENV_ACTIVATE) && $(BLACK) src/ tests/

lint:
	@echo "🔍 Running linting checks..."
	$(VENV_ACTIVATE) && $(FLAKE8) src/ tests/
	$(VENV_ACTIVATE) && $(ISORT) --check-only --diff src/ tests/
	$(VENV_ACTIVATE) && $(BLACK) --check src/ tests/

type-check:
	@echo "🔍 Running type checking..."
	$(VENV_ACTIVATE) && $(MYPY) src/

# Comprehensive check
check: format lint type-check test
	@echo "✅ All checks passed!"

# Pre-commit hooks
pre-commit:
	@echo "🪝 Setting up pre-commit hooks..."
	$(VENV_ACTIVATE) && pre-commit install
	$(VENV_ACTIVATE) && pre-commit run --all-files

# Build and distribution
build:
	@echo "📦 Building distribution packages..."
	$(VENV_ACTIVATE) && $(PYTHON) -m build

# Cleanup
clean:
	@echo "🧹 Cleaning up generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name "*~" -delete 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Development workflow shortcuts
dev-check: format lint type-check
	@echo "🔍 Development checks complete!"

quick-test:
	@echo "⚡ Running quick tests..."
	$(VENV_ACTIVATE) && $(PYTEST) tests/ -x --ff

# Environment validation
validate-env:
	@echo "🔍 Validating environment..."
	$(VENV_ACTIVATE) && $(PYTHON) -c "from src.config import validate_environment; result = validate_environment(); print('Environment validation:', result); exit(0 if all(result.values()) else 1)"

# Documentation (placeholder for future)
docs:
	@echo "📚 Documentation generation not yet implemented"
	@echo "    This will be added in future phases"

# CLI Testing
test-cli:
	@echo "🧪 Testing CLI functionality..."
	$(VENV_ACTIVATE) && $(PYTHON) -m src.cli.main --help

test-cli-commands:
	@echo "🧪 Testing CLI commands..."
	$(VENV_ACTIVATE) && $(PYTHON) -m src.cli.main version
	$(VENV_ACTIVATE) && $(PYTHON) -m src.cli.main list-approaches
	$(VENV_ACTIVATE) && $(PYTHON) -m src.cli.main validate-env

# Experiment shortcuts
run-sample:
	@echo "🧪 Running sample experiment..."
	$(VENV_ACTIVATE) && $(PYTHON) -m src.cli.main run --approach ChainOfThought --samples 5

# Debug and troubleshooting
debug-imports:
	@echo "🔍 Testing all imports..."
	$(VENV_ACTIVATE) && $(PYTHON) -c "import src.config; print('✅ src.config')"
	$(VENV_ACTIVATE) && $(PYTHON) -c "import src.utils.logging_config; print('✅ src.utils.logging_config')"
	$(VENV_ACTIVATE) && $(PYTHON) -c "import config; print('✅ config (backward compatibility)')"
	$(VENV_ACTIVATE) && $(PYTHON) -c "from src.reasoning import get_available_approaches; print('✅ src.reasoning - Available approaches:', get_available_approaches())"
	$(VENV_ACTIVATE) && $(PYTHON) -c "from src.core.reasoning_inference import ReasoningInference; print('✅ src.core.reasoning_inference')"
	$(VENV_ACTIVATE) && $(PYTHON) -c "from src.core.experiment_runner import ExperimentRunner; print('✅ src.core.experiment_runner')"

show-deps:
	@echo "📦 Installed packages:"
	$(VENV_ACTIVATE) && $(UV) pip list

show-config:
	@echo "⚙️  Current configuration:"
	$(VENV_ACTIVATE) && $(PYTHON) -c "from src.config import get_default_config; print(get_default_config().to_dict())"

# CI/CD helpers
ci-install:
	$(UV) pip install -e ".[dev]"

ci-test:
	$(PYTEST) tests/ --cov=src --cov-report=xml --cov-fail-under=80

ci-lint:
	$(FLAKE8) src/ tests/
	$(BLACK) --check src/ tests/
	$(ISORT) --check-only src/ tests/
	$(MYPY) src/

# Version information
version:
	@echo "ML Agents Project Information"
	@echo "=============================="
	$(VENV_ACTIVATE) && $(PYTHON) -c "import src; print(f'Version: {src.__version__}')"
	@echo "Python: $$($(VENV_ACTIVATE) && python --version)"
	@echo "UV: $$(uv --version)"

# All-in-one development setup
dev-setup: clean install-dev pre-commit test
	@echo "🎉 Development environment fully set up and tested!"

# Help for specific areas
help-testing:
	@echo "Testing Commands Help"
	@echo "===================="
	@echo "make test          - Run unit tests (excludes integration)"
	@echo "make test-cov      - Run unit tests with coverage report"
	@echo "make test-integration - Run integration tests (real API calls)"
	@echo "make test-all      - Run all tests including integration"
	@echo "make test-config   - Test only configuration module"
	@echo "make test-logging  - Test only logging module"
	@echo "make test-backward - Test backward compatibility"
	@echo "make quick-test    - Run tests with fail-fast"

help-quality:
	@echo "Code Quality Commands Help"
	@echo "========================="
	@echo "make format     - Auto-format code with black and isort"
	@echo "make lint       - Check code style and imports"
	@echo "make type-check - Run mypy type checking"
	@echo "make check      - Run all quality checks"
	@echo "make pre-commit - Set up and run pre-commit hooks"
