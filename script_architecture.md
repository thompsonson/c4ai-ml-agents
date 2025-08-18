# BBEH Experiment Script Architecture

## Class Definitions

### ExperimentConfig
```python
class ExperimentConfig:
    """Stores all experiment configuration settings."""

    def __init__(self):
        self.dataset_name: str = "MrLight/bbeh-eval"
        self.sample_count: int = 50
        self.provider: str = "openrouter"
        self.model: str = "openai/gpt-oss-20b:free"
        self.temperature: float = 0.3
        self.max_tokens: int = 512
        self.top_p: float = 0.9
        self.reasoning_approaches: list[str] = ["None", "Chain-of-Thought (CoT)"]
        self.output_dir: str = "./outputs"

    def from_args(self, args) -> None:
        """Update config from command line arguments."""
        pass
```

### BBEHDatasetLoader
```python
class BBEHDatasetLoader:
    """Handles loading and sampling from the BBEH dataset."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset = None

    def load_dataset(self) -> pd.DataFrame:
        """Load dataset from HuggingFace."""
        pass

    def sample_data(self, n: int) -> pd.DataFrame:
        """Return first n samples from dataset."""
        pass

    def validate_format(self) -> bool:
        """Ensure dataset has required columns."""
        pass
```

### ReasoningInference
```python
class ReasoningInference:
    """Core inference engine for reasoning approaches."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.api_keys = {}

    def load_api_keys(self) -> None:
        """Load API keys from config.py."""
        pass

    def run_inference(self, prompt: str) -> tuple[str, float, float]:
        """Execute single inference call."""
        # Returns: (output, duration, cost)
        pass

    def execute_reasoning_pipeline(self, prompt: str, reasoning_type: str) -> dict:
        """Run full reasoning pipeline for given approach."""
        # Returns: {output, duration, cost, trace}
        pass
```

### ExperimentRunner
```python
class ExperimentRunner:
    """Orchestrates experiment execution."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.dataset_loader = BBEHDatasetLoader(config.dataset_name)
        self.inference = ReasoningInference(config)
        self.results = []

    def run_single_experiment(self, reasoning_approach: str) -> pd.DataFrame:
        """Run experiment with single reasoning approach."""
        pass

    def run_comparison(self) -> dict[str, pd.DataFrame]:
        """Run all configured reasoning approaches."""
        pass

    def handle_rate_limits(self, func, *args, **kwargs):
        """Retry logic for API rate limits."""
        pass
```

### ResultsProcessor
```python
class ResultsProcessor:
    """Handles result analysis and output generation."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def save_to_csv(self, results: pd.DataFrame, experiment_name: str) -> str:
        """Save results to timestamped CSV file."""
        pass

    def generate_summary(self, results: pd.DataFrame) -> dict:
        """Calculate summary statistics."""
        # Returns: {avg_time, avg_tokens, samples_count, etc}
        pass

    def compare_approaches(self, results_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create comparison table between approaches."""
        pass

    def create_report(self, results_dict: dict[str, pd.DataFrame]) -> str:
        """Generate markdown summary report."""
        pass
```

### Main Script Structure
```python
def main():
    """Entry point for the script."""
    # 1. Parse command line arguments
    # 2. Create config
    # 3. Initialize components
    # 4. Run experiment(s)
    # 5. Process and save results
    # 6. Generate report
    pass

if __name__ == "__main__":
    main()
```
