# Configuration for Telegram bot using OpenAI models

TELEGRAM_TOKEN = "YOUR_TELEGRAM_TOKEN"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
MODEL = "gpt-4o"
CONTEXT_SIZE = 4096
CONTEXT_WINDOW_MESSAGES = 20
DOC_MAX_CHARS = 2000
TELEGRAM_MAX_CHARS = 4096

# List of assistants interacting with each other.
# Each assistant has a role and a system prompt.
ASSISTANTS = [
    {"role": "assistant", "system_prompt": "You are a helpful assistant."},
]

# Assistants used when the user starts a medical consilium. The first agent acts
# as a doctor providing advice, and the second agent critiques the doctor's
# conclusions.
CONSILIUM_ASSISTANTS = [
    {
        "role": "doctor",
        "system_prompt": (
            "You are an experienced medical doctor. Provide detailed clinical "
            "opinions and suggestions based on the user's questions."
        ),
    },
    {
        "role": "critic",
        "system_prompt": (
            "You critically evaluate the doctor's response. Point out "
            "potential issues or missing considerations and suggest "
            "improvements."
        ),
    },
]
