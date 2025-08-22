# Claude Code Instructions for ML Agents Project

This file contains specific instructions for Claude Code when working with the ML Agents reasoning research repository.

## Repository Context

This is a Cohere Labs Open Science Research initiative focused on investigating the efficacy of reasoning in AI models. The project explores how different reasoning approaches (Chain of Thought, Program of Thought, etc.) impact model performance across various tasks.

**Core Research Questions:**
1. Do all tasks benefit from reasoning?
2. Do different models show varying benefits from reasoning?
3. How do different reasoning approaches compare?
4. Is there a task-approach fit?
5. What are the cost-benefit tradeoffs?
6. Can we predict reasoning needs from prompts?

## Project Structure

```
ml-agents/
├── Reasoning_LLM.ipynb      # Main Jupyter notebook (to be refactored)
├── config.py                # API key management and validation
├── requirements.txt         # Python dependencies
├── setup.sh                # Environment setup script
├── script_architecture.md   # CLI app architecture plan
├── EXAMPLE_EXPERIMENT.md    # Step-by-step experiment guide
├── documentation/           # CSV files with benchmarks and tracking
│   ├── Tasks - Benchmarks.csv
│   └── ML Agents Task Tracker.csv
└── .env                    # API keys (not in git)
```

## Development Process

### 1. **Stringent Testing Requirements**
- Write unit tests for all new classes and functions
- Test coverage should be maintained above 80%
- Mock all external API calls in tests
- Validate all reasoning pipelines with sample data
- Test error handling and edge cases

### 2. **Code Quality Standards**
- Use type hints for all function parameters and returns
- Follow PEP 8 style guidelines
- Document all classes and methods with docstrings
- Handle exceptions gracefully with proper error messages
- Use logging instead of print statements

### 3. **Architecture Guidelines**

Follow the class structure defined in `script_architecture.md`:

```python
# Core Classes
- ExperimentConfig: Configuration management
- BBEHDatasetLoader: Dataset loading and validation
- ReasoningInference: Core inference engine
- ExperimentRunner: Experiment orchestration
- ResultsProcessor: Result analysis and output
```

### 4. **API Integration Best Practices**
- Always validate API keys before use
- Implement exponential backoff for rate limits
- Log all API calls with timestamps
- Track costs per API call
- Cache model responses where appropriate

## Common Commands

### Environment Setup
```bash
# Initial setup
./setup.sh

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Running Experiments
```bash
# Run Jupyter notebook (current)
jupyter notebook Reasoning_LLM.ipynb

# Future CLI usage
ml-agents run --provider openrouter --model gpt-3.5-turbo --approach ChainOfThought --samples 50
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_experiment_runner.py
```

## File Structure for CLI App

```
ml-agents/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── experiment_config.py
│   │   ├── dataset_loader.py
│   │   ├── reasoning_inference.py
│   │   ├── experiment_runner.py
│   │   └── results_processor.py
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── chain_of_thought.py
│   │   ├── program_of_thought.py
│   │   ├── reasoning_as_planning.py
│   │   ├── reflection.py
│   │   ├── chain_of_verification.py
│   │   ├── skeleton_of_thought.py
│   │   ├── tree_of_thought.py
│   │   ├── graph_of_thought.py
│   │   ├── rewoo.py
│   │   └── buffer_of_thoughts.py
│   └── utils/
│       ├── __init__.py
│       ├── api_clients.py
│       ├── rate_limiter.py
│       └── logging_config.py
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_dataset_loader.py
│   ├── test_reasoning_inference.py
│   ├── test_experiment_runner.py
│   ├── test_results_processor.py
│   └── test_reasoning/
│       └── test_*.py
├── outputs/
│   └── [timestamped result files]
├── requirements.txt
├── requirements-dev.txt     # Development dependencies
├── pytest.ini              # Pytest configuration
├── .env.example            # Example environment variables
└── README.md
```

## Implementation Priorities

1. **Phase 1: Core Infrastructure**
   - Set up project structure
   - Implement ExperimentConfig with CLI argument parsing
   - Create base classes for reasoning approaches
   - Set up logging and error handling

2. **Phase 2: Data and API Integration**
   - Implement BBEHDatasetLoader with validation
   - Create API client wrappers with rate limiting
   - Add configuration loading from environment

3. **Phase 3: Reasoning Pipelines**
   - Implement each reasoning approach as a separate class
   - Create ReasoningInference to orchestrate approaches
   - Add comprehensive error handling

4. **Phase 4: Experiment Execution**
   - Build ExperimentRunner with parallel execution support
   - Implement progress tracking and resumption
   - Add result caching

5. **Phase 5: Results and Analysis**
   - Create ResultsProcessor with multiple output formats
   - Add comparison and visualization capabilities
   - Generate comprehensive reports

## Testing Strategy

### Unit Tests
- Test each class in isolation
- Mock all external dependencies
- Validate edge cases and error conditions

### Integration Tests
- Test reasoning pipelines end-to-end
- Validate API integrations with test keys
- Check data flow through the system

### Performance Tests
- Measure inference times
- Track memory usage
- Validate concurrent execution

## Important Conventions

### Naming Conventions
- Classes: PascalCase (e.g., `ExperimentRunner`)
- Functions/methods: snake_case (e.g., `run_inference`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_RETRIES`)
- Private methods: prefix with underscore (e.g., `_validate_config`)

### Result File Naming
```
{provider}_{model}_{reasoning_approach}_{timestamp}.csv
```

### Error Messages
- Be specific about what went wrong
- Include relevant context (model, reasoning approach, etc.)
- Suggest corrective actions when possible

## When Making Changes

1. **Before Implementation:**
   - Review script_architecture.md for design decisions
   - Check existing code patterns in the notebook
   - Validate approach with small test cases

2. **During Implementation:**
   - Write tests alongside code
   - Use type hints consistently
   - Add comprehensive logging
   - Handle errors gracefully

3. **After Implementation:**
   - Run full test suite
   - Update documentation
   - Test with multiple providers/models
   - Validate output formats

## Troubleshooting

### Common Issues
- **API Key Errors**: Check .env file and validate with config.py
- **Rate Limits**: Implement exponential backoff
- **Memory Issues**: Process datasets in chunks
- **Import Errors**: Ensure virtual environment is activated

### Debugging Tips
- Use logging at DEBUG level for detailed traces
- Save intermediate results for analysis
- Test with small datasets first
- Validate API responses before processing

## Future Enhancements

- Add support for batch processing
- Implement result caching and resumption
- Create web interface for experiment management
- Add real-time monitoring dashboard
- Support for custom reasoning approaches
- Integration with MLflow for experiment tracking

## MCP Integration (Development Tool)

The project includes **SQLite database persistence** for all experiment results and supports **Claude Code MCP server integration** for direct database queries during development and analysis.

### Database Features
- **Real-time persistence**: All experiment results are automatically saved to `ml_agents_results.db`
- **Read-only MCP access**: Query the database directly from Claude Code conversations
- **Development workflow**: Enhanced debugging and analysis capabilities

### MCP Server Setup (Development)

For developers using Claude Code, enable direct database queries:

```bash
# Configure MCP server (one-time setup)
make configure-mcp

# Or run the script directly
./scripts/install-sqlite-mcp-server.sh
```

**Available MCP Tools**:
- `read_query`: Execute validated SELECT queries
- `list_tables`: Show all database tables
- `describe_table`: Show table schemas

**Development Workflow**:
1. Run experiments using CLI or Jupyter notebook
2. Results are automatically persisted to SQLite database
3. Use Claude Code with MCP server to query and analyze results
4. Export data in multiple formats (CSV, JSON, Excel)

**⚠️ Note**: Project-scoped MCP servers don't appear in `claude mcp list` due to a [known bug](https://github.com/anthropics/claude-code/issues/5963). Use `claude mcp get sqlite-read-only` to verify installation.

### Database CLI Commands

```bash
# Database management
ml-agents db-init --db-path ./results.db          # Initialize database
ml-agents db-backup --source ./results.db         # Create backup
ml-agents db-stats --db-path ./results.db         # Show statistics

# Export and analysis
ml-agents export EXPERIMENT_ID --format excel     # Export to Excel
ml-agents compare-experiments "exp1,exp2,exp3"    # Compare experiments
ml-agents analyze EXPERIMENT_ID --type accuracy   # Generate reports
ml-agents list-experiments --status completed     # List experiments
```

## Community Collaboration

- Share results in documentation/Tasks - Benchmarks.csv
- Update ML Agents Task Tracker with progress
- Coordinate experiments via Discord #ml-agents channel
- Follow Cohere Labs contribution guidelines

Remember: The goal is to create a robust, extensible platform for reasoning research that the community can build upon. Prioritize code quality, comprehensive testing, and clear documentation.
