# Project Structure

**Version:** 1.0
**Date:** 2025-09-17
**Purpose:** Define codebase organization following DDD principles

## Directory Structure

```
ml_agents_v2/
├── pyproject.toml              # uv project configuration
├── uv.lock                     # Dependency lockfile
├── .env.example                # Environment template
├── .gitignore                  # Git exclusions
├── README.md                   # Project documentation
├── alembic.ini                 # Database migration config
├── alembic/                    # Migration scripts
│   └── versions/
├── src/
│   └── ml_agents_v2/
│       ├── __init__.py
│       ├── main.py             # CLI entry point
│       ├── config/
│       │   ├── __init__.py
│       │   └── application_config.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── domain/         # Domain layer (business logic)
│       │   │   ├── __init__.py
│       │   │   ├── entities/
│       │   │   │   ├── __init__.py
│       │   │   │   ├── evaluation.py
│       │   │   │   └── preprocessed_benchmark.py
│       │   │   ├── value_objects/
│       │   │   │   ├── __init__.py
│       │   │   │   ├── agent_config.py
│       │   │   │   ├── question.py
│       │   │   │   ├── answer.py
│       │   │   │   ├── evaluation_results.py
│       │   │   │   ├── reasoning_trace.py
│       │   │   │   └── failure_reason.py
│       │   │   ├── services/
│       │   │   │   ├── __init__.py
│       │   │   │   ├── reasoning_agent_service.py
│       │   │   │   ├── reasoning_agent_factory.py
│       │   │   │   └── reasoning/
│       │   │   │       ├── __init__.py
│       │   │   │       ├── none_agent.py
│       │   │   │       └── chain_of_thought_agent.py
│       │   │   └── repositories/
│       │   │       ├── __init__.py
│       │   │       ├── evaluation_repository.py
│       │   │       └── benchmark_repository.py
│       │   └── application/    # Application services
│       │       ├── __init__.py
│       │       ├── services/
│       │       │   ├── __init__.py
│       │       │   ├── evaluation_orchestrator.py
│       │       │   └── benchmark_processor.py
│       │       └── dto/
│       │           ├── __init__.py
│       │           ├── evaluation_info.py
│       │           └── progress_info.py
│       ├── infrastructure/     # Infrastructure layer
│       │   ├── __init__.py
│       │   ├── database/
│       │   │   ├── __init__.py
│       │   │   ├── models.py   # SQLAlchemy models
│       │   │   ├── repositories/
│       │   │   │   ├── __init__.py
│       │   │   │   ├── evaluation_repository_impl.py
│       │   │   │   └── benchmark_repository_impl.py
│       │   │   └── migrations/
│       │   ├── openrouter/
│       │   │   ├── __init__.py
│       │   │   ├── client.py
│       │   │   ├── error_mapper.py
│       │   │   └── model_registry.py
│       │   └── container.py    # Dependency injection
│       └── cli/                # CLI interface
│           ├── __init__.py
│           ├── main.py         # Click command groups
│           ├── commands/
│           │   ├── __init__.py
│           │   ├── evaluate.py
│           │   ├── benchmark.py
│           │   └── health.py
│           └── utils/
│               ├── __init__.py
│               ├── formatting.py
│               ├── progress.py
│               └── error_handling.py
└── tests/
    ├── __init__.py
    ├── conftest.py             # Pytest configuration
    ├── fixtures/               # Test data
    │   ├── benchmarks/
    │   └── evaluations/
    ├── unit/                   # Unit tests
    │   ├── domain/
    │   ├── application/
    │   ├── infrastructure/
    │   └── cli/
    ├── integration/            # Integration tests
    │   ├── database/
    │   ├── openrouter/
    │   └── cli/
    └── acceptance/             # End-to-end tests
        └── test_evaluation_workflow.py
```

## Layer Dependencies

```
CLI → Application → Domain ← Infrastructure
```

**Dependency Rules:**

- CLI depends on Application services
- Application depends on Domain interfaces
- Infrastructure implements Domain interfaces
- Domain has no external dependencies

## Package Responsibilities

### `core/domain/`

Pure business logic with no external dependencies.

**entities/**: Aggregate roots with business rules and lifecycle management
**value_objects/**: Immutable objects representing domain concepts
**services/**: Stateless domain operations including reasoning agent implementations
**repositories/**: Interfaces for data persistence

**reasoning/**: Concrete reasoning agent implementations (NoneAgent, ChainOfThoughtAgent) containing pure reasoning logic

### `core/application/`

Orchestrates domain operations and external interactions.

**services/**: Coordinate domain entities and infrastructure
**dto/**: Data transfer objects for application boundaries

### `infrastructure/`

External system integrations and technical implementations.

**database/**: SQLAlchemy models and repository implementations
**openrouter/**: API client, error mapping, and model registry
**container.py**: Dependency injection configuration

### `cli/`

User interface and command handling.

**commands/**: Click command implementations
**utils/**: CLI formatting and error handling

## Import Conventions

```python
# Domain imports (no external dependencies)
from ml_agents_v2.core.domain.entities import Evaluation
from ml_agents_v2.core.domain.value_objects import AgentConfig

# Application imports
from ml_agents_v2.core.application.services import EvaluationOrchestrator

# Infrastructure imports
from ml_agents_v2.infrastructure.database.models import EvaluationModel
from ml_agents_v2.infrastructure.openrouter import OpenRouterClient

# CLI imports
from ml_agents_v2.cli.commands import evaluate
```

## Configuration Management

### Environment Files

- `.env` - Local development (not committed)
- `.env.example` - Template with dummy values
- `.env.test` - Test environment configuration

### Config Loading

```python
# config/application_config.py
from pydantic_settings import BaseSettings

class ApplicationConfig(BaseSettings):
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

## Database Organization

### Migration Scripts

```
alembic/versions/
├── 001_initial_schema.py
├── 002_add_failure_reason.py
└── 003_add_indexes.py
```

### Model Mapping

- Domain entities → SQLAlchemy models in `infrastructure/database/models.py`
- Repository interfaces → Implementations in `infrastructure/database/repositories/`

## Entry Points

### CLI Entry Point

```python
# main.py
from ml_agents_v2.cli.main import cli

if __name__ == "__main__":
    cli()
```

### Package Entry Point

```toml
# pyproject.toml
[project.scripts]
ml-agents = "ml_agents_v2.cli.main:cli"
```

## Development Tools

### Code Quality

- **black**: Code formatting
- **ruff**: Linting and import sorting
- **mypy**: Type checking

### Testing

- **pytest**: Test framework
- **pytest-asyncio**: Async test support
- **pytest-mock**: Mocking utilities
- **click.testing**: CLI testing

### Build Tools

- **uv**: Package management and virtual environments
- **alembic**: Database migrations

## See Also

- **[Domain Model](v2-domain-model.md)** - Business entities organized in the domain layer
- **[Infrastructure Requirements](v2-infrastructure-requirements.md)** - Technology dependencies and setup requirements
- **[Application Services Architecture](v2-application-services-architecture.md)** - Service layer coordination patterns
- **[CLI Design](v2-cli-design.md)** - CLI implementation structure and organization
- **[Testing Strategy](v2-testing-strategy.md)** - Test organization following this project structure
