# Configuration for Telegram bot using OpenAI models

TELEGRAM_TOKEN = "YOUR_TELEGRAM_TOKEN"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
MODEL = "gpt-4o"
CONTEXT_SIZE = 4096
CONTEXT_WINDOW_MESSAGES = 20
DOC_MAX_CHARS = 2000

# List of assistants interacting with each other.
# Each assistant has a role and a system prompt.
ASSISTANTS = [
    {"role": "assistant", "system_prompt": "You are a helpful assistant."},
]
