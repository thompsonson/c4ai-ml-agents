# Cohere Labs Open Science Research into ML Agents and Reasoning

## Community Resources

- **[ML Agents Community Program](https://sites.google.com/cohere.com/coherelabs-community/community-programs/ml-agents)** - Main hub for Cohere Labs' community-driven initiative on open-source agent research, focusing on agentic frameworks, applications, evaluations, and benchmarks

- **[Project Documentation](https://docs.google.com/document/d/1fLnwUzTvO3XuvViBwLz-QuSe_y87a1p4j8Uw2R4eBiI/edit?pli=1&tab=t.0#heading=h.d0279byf6lhr)** - Detailed specifications and roadmap for the ZeroHPO (Zero-shot Hyperparameter Optimization) project for agentic tasks

- **[Project Tracker](https://docs.google.com/spreadsheets/d/1-TBlPSIiBymQfCdF_LCYJznLwaxcKtYTRJME0NT17kU/edit?usp=sharing)** - Community project tracking, task assignments, and progress monitoring

- **[Discord Community](https://discord.gg/ckaQnUakYx)** - Join the #ml-agents channel for discussions, meetings, and collaboration with the community


## Auto-generate description

[Auto-generated from the notebook]

This project investigates how different reasoning approaches impact AI model performance across various tasks. It provides a comprehensive framework for comparing 10 different reasoning techniques with multiple language models.

## Research Questions

1. **Universal Benefit**: Do all tasks benefit from reasoning?
2. **Model Variability**: Do different models show varying benefits from reasoning?
3. **Approach Comparison**: How do different reasoning approaches (CoT, PoT, etc.) compare?
4. **Task-Approach Fit**: Do certain tasks benefit more from specific reasoning methods?
5. **Cost-Benefit Analysis**: What is the tradeoff for each approach and task?
6. **Predictive Reasoning**: Can we predict the need for reasoning based on the input prompt alone?

## Reasoning Approaches Implemented

1. **None** - Direct prompting without reasoning
2. **Chain-of-Thought (CoT)** - Step-by-step reasoning
3. **Program-of-Thought (PoT)** - Python program generation for problem-solving
4. **Reasoning-as-Planning (RAP)** - Planning-based approach with action sequences
5. **Reflection** - Draft, critique, and improve cycle
6. **Chain-of-Verification (CoVe)** - Verification questions to check accuracy
7. **Skeleton-of-Thought (SoT)** - Outline generation followed by expansion
8. **Tree-of-Thought (ToT)** - Multiple approaches evaluated and best selected
9. **Graph-of-Thought (GoT)** - Knowledge graph synthesis approach
10. **ReWOO** - Plan with tool use simulation
11. **Buffer-of-Thoughts (BoT)** - Multi-step reasoning with thought buffer

## Setup

### Prerequisites

- Python 3.8+
- uv (for virtual environment management)
- API keys for at least one provider (Anthropic, Cohere, OpenRouter, or Hugging Face)

### Quick Start

1. Clone the repository and navigate to the project directory

2. Run the setup script:
   ```bash
   ./setup.sh
   ```

3. Add your API keys to the `.env` file:
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

4. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

5. Launch the notebook:
   ```bash
   jupyter notebook Reasoning_LLM.ipynb
   ```

### Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

### Supported Providers and Models

- **Hugging Face**: Gemma-2, Mistral-7B, Llama-3-8B
- **Anthropic**: Claude Opus 4, Claude Sonnet 4, Claude 3.5 Haiku
- **Cohere**: Command R+, Command R, Command Light
- **OpenRouter**: GPT-5, GPT-5 Mini, GPT OSS-120B, Gemini 2.5 Flash Lite

### Hyperparameters

- **Temperature**: 0.0 - 2.0 (controls randomness)
- **Max Tokens**: 64 - 4096 (output length limit)
- **Top P**: 0.0 - 1.0 (nucleus sampling parameter)

## Usage

1. **Setup**: Run the first cell to install libraries (already done if using setup.sh)

2. **Configuration**: Use the interactive widgets to select:
   - Provider and model
   - Hyperparameters (temperature, max tokens, top_p)
   - Reasoning approach

3. **Data**: The notebook loads the "bbeh-eval" dataset by default. Replace with your own Hugging Face dataset:
   ```python
   user_name = "your_username"
   dataset_name = "your_dataset"
   ```

4. **Execute**: Run the experiment cells to process your dataset

5. **Results**: Results are displayed in a table and saved to CSV with filename format:
   `{model_name}_{reasoning_approach}_{timestamp}.csv`

## Dataset Requirements

Your dataset should include:
- **input** column: The question/problem to solve
- **answer** column (optional): Expected output for evaluation
- **task** column (optional): Task category for analysis

## Output Files

The notebook generates CSV files containing:
- Input prompts
- Model outputs
- Full reasoning traces
- Execution time
- Cost estimates
- Configuration details

## Project Structure

```
ml-agents/
├── Reasoning_LLM.ipynb    # Main notebook
├── config.py              # Environment variable loader
├── requirements.txt       # Python dependencies
├── setup.sh              # Automated setup script
├── .env                  # API keys (create from .env.example)
├── .env.example          # Template for API keys
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Tips

1. Start with a small dataset subset for testing
2. Use lower temperatures (0.3-0.7) for more consistent results
3. Monitor API costs, especially with large datasets
4. Save intermediate results frequently
5. The notebook caches Hugging Face models to avoid reloading

## Troubleshooting

- **GPU Memory**: If using local Hugging Face models, ensure sufficient GPU memory
- **API Limits**: Be aware of rate limits for API providers
- **Missing Keys**: The setup script will warn about missing API keys
- **Dataset Loading**: Ensure your Hugging Face dataset is public or you're authenticated

## Contributing

Feel free to extend the notebook with:
- Additional reasoning approaches
- New evaluation metrics
- Support for more models/providers
- Performance optimizations

## License

### Recommended: CC BY 4.0 (Creative Commons Attribution 4.0 International)

This project is licensed under the Creative Commons Attribution 4.0 International License. This means:

- ✅ **Share** - Copy and redistribute the material in any medium or format
- ✅ **Adapt** - Remix, transform, and build upon the material for any purpose, even commercially
- ✅ **Attribution** - You must give appropriate credit, provide a link to the license, and indicate if changes were made

This license is chosen because:
1. **Open Science**: Aligns with Cohere Labs' open science mission
2. **Maximum Impact**: Allows both academic and commercial use, accelerating AI research
3. **Community Growth**: Enables derivatives while ensuring original work is credited
4. **Simplicity**: Easy to understand and implement

**Note**: For the code components specifically, you may want to consider dual-licensing with MIT or Apache 2.0 for better software compatibility.

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a>

### Alternative Options Considered:

- **CC BY-SA 4.0**: Adds "ShareAlike" requirement - derivatives must use same license (more restrictive but ensures openness)
- **CC BY-NC 4.0**: Adds "NonCommercial" restriction - prevents commercial use (limits industry collaboration)
- **CC0**: Public domain dedication - no attribution required (maximum freedom but no credit requirement)