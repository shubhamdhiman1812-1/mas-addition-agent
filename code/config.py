
# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    """
    Configuration management for General Addition Assistant.
    Handles environment variable loading, API key management,
    LLM configuration, domain settings, validation, and fallbacks.
    """

    # --- API Key Management ---
    @staticmethod
    def get_openai_api_key() -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Raise clear error if API key is missing
            raise ConfigError("OPENAI_API_KEY is missing. Please set it in your environment or .env file.")
        return api_key

    # --- LLM Configuration ---
    @staticmethod
    def get_llm_provider() -> str:
        return os.getenv("LLM_PROVIDER", "openai")

    @staticmethod
    def get_llm_model() -> str:
        return os.getenv("OPENAI_MODEL", "gpt-4.1")

    @staticmethod
    def get_llm_temperature() -> float:
        try:
            return float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        except Exception:
            return 0.7

    @staticmethod
    def get_llm_max_tokens() -> int:
        try:
            return int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
        except Exception:
            return 2000

    @staticmethod
    def get_llm_system_prompt() -> str:
        return (
            "You are a helpful assistant whose role is to help users add two numbers. "
            "When a user provides two numbers, validate that both are numeric values. "
            "If both inputs are valid, perform the addition and return the result in a friendly, conversational tone. "
            "If the input is invalid or missing, politely prompt the user to provide two valid numbers. "
            "Never perform operations other than addition. If you cannot process the request, respond with a clear and supportive fallback message."
        )

    @staticmethod
    def get_llm_user_prompt_template() -> str:
        return "Please enter two numbers you'd like to add together."

    @staticmethod
    def get_llm_few_shot_examples() -> list:
        return [
            "Add 3 and 4 -> The sum of 3 and 4 is 7.",
            "What is 10 plus 15? -> The sum of 10 and 15 is 25."
        ]

    # --- Domain-Specific Settings ---
    @staticmethod
    def get_domain() -> str:
        return os.getenv("AGENT_DOMAIN", "general")

    @staticmethod
    def get_agent_name() -> str:
        return os.getenv("AGENT_NAME", "General Addition Assistant")

    @staticmethod
    def get_personality() -> str:
        return os.getenv("AGENT_PERSONALITY", "casual")

    # --- Validation and Error Handling ---
    @staticmethod
    def validate_config():
        # Validate required API key
        Config.get_openai_api_key()
        # Add more validation as needed

    # --- Default Values and Fallbacks ---
    @staticmethod
    def get_fallback_response() -> str:
        return "Sorry, I couldn't process your input. Please provide two numbers, and I'll add them for you!"

    @staticmethod
    def get_output_format() -> str:
        return 'Respond with a short, friendly sentence stating the sum. Example: "The sum of 5 and 7 is 12."'

    # --- Utility: All LLM Config as Dict ---
    @staticmethod
    def get_llm_config() -> dict:
        return {
            "provider": Config.get_llm_provider(),
            "model": Config.get_llm_model(),
            "temperature": Config.get_llm_temperature(),
            "max_tokens": Config.get_llm_max_tokens(),
            "system_prompt": Config.get_llm_system_prompt(),
            "user_prompt_template": Config.get_llm_user_prompt_template(),
            "few_shot_examples": Config.get_llm_few_shot_examples(),
        }

    # --- Utility: All Domain Config as Dict ---
    @staticmethod
    def get_domain_config() -> dict:
        return {
            "domain": Config.get_domain(),
            "agent_name": Config.get_agent_name(),
            "personality": Config.get_personality(),
            "output_format": Config.get_output_format(),
            "fallback_response": Config.get_fallback_response(),
        }

# Example usage (uncomment for testing):
# try:
#     Config.validate_config()
#     print("Config loaded successfully.")
# except ConfigError as e:
#     print(f"Configuration error: {e}")
