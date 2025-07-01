"""Telegram bot that uses OpenAI models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Union, Iterable
import base64
from types import SimpleNamespace

try:  # Allow running tests without installed packages
    import openai
except Exception:  # pragma: no cover - handled in tests
    openai = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: None)))

try:
    from telegram import Update
    from telegram.ext import (
        ApplicationBuilder,
        CommandHandler,
        MessageHandler,
        filters,
        CallbackContext,
    )
except Exception:  # pragma: no cover - telegram not available in tests
    Update = CallbackContext = object
    filters = SimpleNamespace(ALL=None)
    ApplicationBuilder = CommandHandler = MessageHandler = SimpleNamespace

# Allow running the script directly by ensuring the package root is on sys.path
from pathlib import Path
import sys

package_root = Path(__file__).resolve().parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

import bot.settings as settings


def _create_chat_completion(**kwargs):
    """Call the correct OpenAI completion method across library versions."""
    if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
        return openai.chat.completions.create(**kwargs)
    return openai.ChatCompletion.create(**kwargs)

logger = logging.getLogger(__name__)


def setup_openai() -> None:
    """Configure OpenAI API key."""
    openai.api_key = settings.OPENAI_API_KEY


@dataclass
class Assistant:
    role: str
    system_prompt: str


class OpenAIBot:
    def __init__(self, assistants: List[Assistant] | None = None) -> None:
        self.assistants = assistants or [
            Assistant(role=a["role"], system_prompt=a["system_prompt"]) for a in settings.ASSISTANTS
        ]
        setup_openai()

    def _query_assistant(self, conversation: List[dict], assistant: Assistant) -> str:
        messages = [{"role": "system", "content": assistant.system_prompt}] + conversation
        logger.debug("Sending messages: %s", messages)
        resp = _create_chat_completion(model=settings.MODEL, messages=messages)
        if isinstance(resp, dict):
            content = resp["choices"][0]["message"]["content"]
        else:
            content = resp.choices[0].message.content
        conversation.append({"role": assistant.role, "content": content})
        return content

    def ask(self, content: Union[str, List[dict]]) -> str:
        conversation = [{"role": "user", "content": content}]
        reply = ""
        for assistant in self.assistants:
            reply = self._query_assistant(conversation, assistant)
        return reply


class TelegramBot:
    def __init__(self) -> None:
        self.application = ApplicationBuilder().token(settings.TELEGRAM_TOKEN).build()
        self.bot = OpenAIBot()

        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(MessageHandler(filters.ALL, self.handle_message))

    async def start(self, update: Update, context: CallbackContext) -> None:
        await update.message.reply_text("Hello! Send me a message and I will respond using OpenAI.")

    async def handle_message(self, update: Update, context: CallbackContext) -> None:
        message = update.message
        text = message.caption or message.text or ""

        # Default content is just the text string
        content: Union[str, List[dict]] = text

        # Handle attached files
        if message.document:
            file_name = message.document.file_name
            if text:
                text += f"\n[User attached file: {file_name}]"
            else:
                text = f"[User attached file: {file_name}]"
            content = text

        if message.photo:
            # Download largest size photo
            photo = message.photo[-1]
            file = await photo.get_file()
            image_bytes: Iterable[int] = await file.download_as_bytearray()
            b64 = base64.b64encode(bytearray(image_bytes)).decode("utf-8")
            image_url = f"data:image/jpeg;base64,{b64}"
            content = []
            if text:
                content.append({"type": "text", "text": text})
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        if message.audio:
            if text:
                text += "\n[User sent an audio file]"
            else:
                text = "[User sent an audio file]"
            content = text

        reply = self.bot.ask(content)
        await message.reply_text(reply)

    def run(self) -> None:
        logger.info("Starting bot")
        self.application.run_polling()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    TelegramBot().run()
