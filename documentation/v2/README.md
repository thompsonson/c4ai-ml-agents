# ML Agents v2 Architecture Documentation

## Overview

**ML Agents v2 is a complete architectural redesign and clean break from v1.** This represents a fundamental reimagining of the reasoning research platform, transitioning from the v1 Jupyter notebook prototype to a production-ready CLI application built from the ground up.

### Clean Break from v1

**v2 is NOT an evolution of v1 - it is a complete rewrite with:**
- **New Architecture**: Domain-Driven Design with clean architecture layers vs. monolithic notebook
- **New Interface**: CLI application vs. Jupyter notebook interface
- **New Data Model**: Structured entities and value objects vs. ad-hoc data handling
- **New Infrastructure**: OpenRouter integration, SQLite persistence, dependency injection vs. direct API calls
- **New Codebase**: Fresh implementation following architectural principles vs. research prototype code

**Migration Strategy**: v2 will be developed as a separate project. v1 remains available for reference but will not be maintained once v2 is production-ready.

### v2 Design Principles

The v2 redesign focuses on Domain-Driven Design principles, clean architecture separation, and robust evaluation workflows for comparing reasoning approaches across diverse benchmarks.

Key architectural decisions include using SQLite for development with PostgreSQL extensibility, OpenRouter for unified LLM provider access, and a synchronous CLI execution model that prioritizes simplicity and real-time progress feedback over complex concurrency management.

The platform supports systematic evaluation of reasoning approaches (None, Chain of Thought, and future extensions) against preprocessed benchmarks, with comprehensive result tracking and failure analysis to support research into task-approach fit and reasoning effectiveness.

## Reading Guide

### For New Team Members (Start Here)
1. **[Domain Model](v2-domain-model.md)** - Core business entities, aggregates, and relationships
2. **[Ubiquitous Language](v2-ubiquitous-language.md)** - Shared vocabulary between researchers and developers
3. **[Core Behaviors](v2-core-behaviour-definition.md)** - Key user workflows and evaluation execution patterns

### For Developers
1. **[Project Structure](v2-project-structure.md)** - Codebase organization following DDD layers
2. **[Agents](v2-agents.md)** - Reasoning agent implementations and behavioral specifications
3. **[Infrastructure Requirements](v2-infrastructure-requirements.md)** - Dependencies, OpenRouter integration, and deployment setup
4. **[CLI Design](v2-cli-design.md)** - Command interface implementation and user interaction patterns

### For Architects
1. **[Application Services Architecture](v2-application-services-architecture.md)** - Service coordination and transaction boundaries
2. **[Data Model](v2-data-model.md)** - Database schema design and persistence strategy
3. **[Testing Strategy](v2-testing-strategy.md)** - Quality assurance approach for research workflows

## Document Index

| Document | Purpose | Prerequisites | Target Audience |
|----------|---------|---------------|-----------------|
| **v2-domain-model.md** | Define core business entities, value objects, and domain relationships | None | All team members |
| **v2-ubiquitous-language.md** | Establish shared vocabulary for evaluation concepts | Domain Model | All team members |
| **v2-core-behaviour-definition.md** | Specify evaluation workflows and user interaction patterns | Domain Model, Ubiquitous Language | Product owners, developers |
| **v2-agents.md** | Define reasoning agent implementations and behavioral specifications | Domain Model, Core Behaviors | Researchers, developers |
| **v2-application-services-architecture.md** | Design service coordination and transaction management | Domain Model, Core Behaviors | Architects, senior developers |
| **v2-cli-design.md** | Define command structure and user interface patterns | Core Behaviors | Developers, UX consideration |
| **v2-data-model.md** | Specify database schema and persistence strategy | Domain Model | Database developers, architects |
| **v2-infrastructure-requirements.md** | Detail external dependencies and deployment configuration | Data Model | DevOps, infrastructure engineers |
| **v2-project-structure.md** | Organize codebase following DDD architectural layers | All architecture docs | Developers |
| **v2-testing-strategy.md** | Define testing approach for research platform requirements | Core Behaviors, Application Services | QA engineers, developers |

## Change Impact Guide

When making architectural changes, consider reviewing these related documents:

| If You Modify... | Also Review... | Reason |
|------------------|----------------|--------|
| **Domain entities** | Data Model, Application Services | Entity changes affect persistence and orchestration |
| **CLI commands** | Core Behaviors, CLI Design | Command changes impact user workflows |
| **Infrastructure dependencies** | Project Structure, Testing Strategy | Infrastructure changes affect build and test configuration |
| **Evaluation workflows** | Application Services, Data Model | Workflow changes impact service coordination and data storage |
| **Reasoning approaches** | Agents, Domain Model, Core Behaviors | New approaches require agent implementation, domain updates, and workflow integration |
| **Database schema** | Infrastructure Requirements, Project Structure | Schema changes may require migration tools and environment updates |

## Document Versions

| Document | Version | Last Updated | Status |
|----------|---------|--------------|--------|
| **v2-domain-model.md** | 1.1 | 2025-09-17 | Complete |
| **v2-ubiquitous-language.md** | 1.0 | 2025-09-17 | Complete |
| **v2-core-behaviour-definition.md** | 1.2 | 2025-09-17 | Complete |
| **v2-agents.md** | 1.1 | 2025-09-17 | Complete |
| **v2-application-services-architecture.md** | 1.0 | 2025-09-17 | Complete |
| **v2-cli-design.md** | 1.0 | 2025-09-17 | Complete |
| **v2-data-model.md** | 1.0 | 2025-09-17 | Complete |
| **v2-infrastructure-requirements.md** | 1.0 | 2025-09-17 | Complete |
| **v2-project-structure.md** | 1.0 | 2025-09-17 | Complete |
| **v2-testing-strategy.md** | 1.1 | 2025-09-17 | Complete |

## Implementation Status

### Core Architecture ✅ **Design Complete**
- [x] Domain model with aggregates and value objects
- [x] Service layer coordination patterns
- [x] Data persistence strategy
- [x] Infrastructure integration approach

### Implementation Components 🚧 **In Development**
- [ ] Domain entity implementations
- [ ] Repository pattern with SQLite
- [ ] OpenRouter client integration
- [ ] CLI command structure
- [ ] Evaluation orchestration service

### Testing Framework 📋 **Planned**
- [ ] Acceptance test scenarios
- [ ] Repository integration tests
- [ ] CLI workflow validation
- [ ] OpenRouter contract tests

## Quick Reference

### Key Architectural Decisions
- **Domain-Driven Design** with clear aggregate boundaries (Evaluation, PreprocessedBenchmark)
- **Synchronous execution** model for simplicity and real-time progress tracking
- **SQLite development** with PostgreSQL production extensibility
- **OpenRouter API** for unified LLM provider access
- **Structured failure taxonomy** for research-focused error analysis

### Critical Workflows
- **Create Evaluation**: Configure reasoning approach for specific benchmark
- **Execute Evaluation**: Run synchronous evaluation with real-time progress
- **List Resources**: Browse available benchmarks and evaluation history

### Cross-Cutting Concerns
- **Configuration**: 12-factor app with environment variable externalization
- **Logging**: Structured logging with configurable output formats
- **Error Handling**: Domain-aware failure reasons with recovery guidance
- **Testing**: Research workflow validation with non-deterministic response handling

---

*This documentation represents the architectural foundation for ML Agents v2 implementation. All documents should be updated synchronously when making significant architectural changes.*
