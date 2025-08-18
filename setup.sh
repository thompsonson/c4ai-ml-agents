#!/bin/bash

echo "Setting up ML Agents Reasoning Project Environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "uv installed successfully."
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
# API Keys for ML Agents Reasoning Project
# Replace 'your_key_here' with your actual API keys

ANTHROPIC_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
EOF
    echo ".env file created."
fi

# Validate API keys
echo ""
echo "Checking API key configuration..."
python -c "
from config import validate_api_keys
keys_status = validate_api_keys()
missing_keys = [k for k, v in keys_status.items() if not v]
if missing_keys:
    print(f'⚠️  Please add your API keys for: {', '.join(missing_keys)} to the .env file')
else:
    print('✅ All API keys are configured!')
"

echo ""
echo "Setup complete! 🎉"
echo ""
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the notebook locally:"
echo "  jupyter notebook Reasoning_LLM.ipynb"
