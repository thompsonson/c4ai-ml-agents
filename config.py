"""Configuration module for loading environment variables."""

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# API Keys dictionary
API_KEYS = {
    'anthropic': os.getenv('ANTHROPIC_API_KEY'),
    'cohere': os.getenv('COHERE_API_KEY'),
    'openrouter': os.getenv('OPENROUTER_API_KEY'),
    'huggingface': os.getenv('HUGGINGFACE_API_KEY')
}

def get_api_key(provider):
    """Get API key for a specific provider.
    
    Args:
        provider: The provider name (anthropic, cohere, openrouter, huggingface)
        
    Returns:
        The API key string or None if not found
    """
    return API_KEYS.get(provider)

def validate_api_keys():
    """Validate that all API keys are set.
    
    Returns:
        Dictionary with provider names as keys and boolean values indicating if key is set
    """
    return {provider: bool(key) and key != 'your_key_here' 
            for provider, key in API_KEYS.items()}