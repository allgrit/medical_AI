"""Telegram bot that uses OpenAI models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Union, Iterable, Optional, Type
import base64
import asyncio
from types import SimpleNamespace
from io import BytesIO

try:
    from pdfminer.high_level import extract_text
except ImportError:  # pragma: no cover - older pdfminer versions
    from pdfminer.high_level import extract_text_to_fp
    from io import StringIO

    def extract_text(file) -> str:
        output = StringIO()
        extract_text_to_fp(file, output)
        return output.getvalue()


import docx
import openpyxl
import xlrd
import mammoth
import lxml.html
import subprocess
import tempfile
import os
import string
import re

try:  # Allow running tests without installed packages
    import openai
except Exception:  # pragma: no cover - handled in tests
    openai = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: None))
    )

try:
    from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
    from telegram.ext import (
        ApplicationBuilder,
        CommandHandler,
        MessageHandler,
        CallbackQueryHandler,
        filters,
        CallbackContext,
    )
except Exception:  # pragma: no cover - telegram not available in tests
    Update = CallbackContext = object
    filters = SimpleNamespace(ALL=None)
    InlineKeyboardMarkup = lambda *a, **k: None
    InlineKeyboardButton = lambda *a, **k: None
    CallbackQueryHandler = lambda *a, **k: None
    ApplicationBuilder = CommandHandler = MessageHandler = SimpleNamespace

# Allow running the script directly by ensuring the package root is on sys.path
from pathlib import Path
import sys

package_root = Path(__file__).resolve().parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

import bot.settings as settings


def _is_chat_model(model: str | None) -> bool:
    """Return True if the model uses the chat completion API."""
    if not model:
        return True
    name = str(model).lower().strip()
    if "gpt-" in name and "-instruct" not in name:
        return True
    if name.startswith("claude"):
        return True
    if name.startswith("o4"):
        return True
    return False


def _messages_to_prompt(messages: List[dict]) -> str:
    """Convert chat messages into a plain prompt."""
    parts = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            parts.append(content.strip())
        elif role == "user":
            parts.append(f"User: {content}")
        else:
            parts.append(f"Assistant: {content}")
    return "\n".join(parts)


def _create_chat_completion(**kwargs):
    """Call the correct OpenAI completion method across library versions.

    The logic first relies on :func:`_is_chat_model` to pick the appropriate
    endpoint. If the call fails with a ``NotFoundError`` indicating the wrong
    endpoint was used, it transparently retries using the alternative API.
    """

    model = kwargs.get("model")

    def call_chat() -> object:
        if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
            return openai.chat.completions.create(**kwargs)
        return openai.ChatCompletion.create(**kwargs)

    def call_completion() -> object:
        params = kwargs.copy()
        if "messages" in params:
            params["prompt"] = _messages_to_prompt(params.pop("messages"))
        if hasattr(openai, "completions"):
            return openai.completions.create(**params)
        return openai.Completion.create(**params)

    use_chat = _is_chat_model(model)

    try:
        return call_chat() if use_chat else call_completion()
    except openai.NotFoundError as e:
        msg = str(e).lower()
        if use_chat and "v1/responses" in msg:
            return call_completion()
        if not use_chat and "chat" in msg:
            return call_chat()
        raise


logger = logging.getLogger(__name__)


def _extract_doc_text(data: bytes) -> str:
    """Naively extract text from a legacy .doc file."""
    try:
        text = data.decode("utf-16le", errors="ignore")
    except Exception:
        text = ""
    if not text.strip():
        text = data.decode("cp1252", errors="ignore")
    text = "".join(ch if ch.isprintable() or ch in "\n\r\t" else " " for ch in text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def setup_openai() -> None:
    """Configure OpenAI API key."""
    openai.api_key = settings.OPENAI_API_KEY


@dataclass
class Assistant:
    role: str
    system_prompt: str


class BaseBot:
    """Abstract bot interface."""

    def __init__(
        self, assistants: List[Assistant | dict] | None = None, model: str | None = None
    ) -> None:
        self.assistants = [
            (
                a
                if isinstance(a, Assistant)
                else Assistant(role=a["role"], system_prompt=a["system_prompt"])
            )
            for a in (assistants or settings.ASSISTANTS)
        ]
        self.model = model or settings.MODEL

    def ask_stream(
        self, content: Union[str, List[dict]], conversation: Optional[List[dict]] = None
    ) -> Iterable[tuple[str, str]]:
        raise NotImplementedError

    def ask(
        self, content: Union[str, List[dict]], conversation: Optional[List[dict]] = None
    ) -> Union[str, List[tuple[str, str]]]:
        raise NotImplementedError


class OpenAIBot(BaseBot):
    def __init__(
        self, assistants: List[Assistant | dict] | None = None, model: str | None = None
    ) -> None:
        super().__init__(assistants, model)
        setup_openai()

    def _query_assistant(self, conversation: List[dict], assistant: Assistant) -> str:
        messages = [
            {"role": "system", "content": assistant.system_prompt}
        ] + conversation
        logger.debug("Sending messages: %s", messages)
        resp = _create_chat_completion(model=self.model, messages=messages)
        if isinstance(resp, dict):
            content = resp["choices"][0]["message"]["content"]
        else:
            content = resp.choices[0].message.content
        conversation.append({"role": assistant.role, "content": content})
        return content

    def ask_stream(
        self, content: Union[str, List[dict]], conversation: Optional[List[dict]] = None
    ) -> Iterable[tuple[str, str]]:
        """Yield replies from assistants sequentially as they are produced."""
        if conversation is None:
            conversation = []
        conversation.append({"role": "user", "content": content})

        for assistant in self.assistants:
            reply = self._query_assistant(conversation, assistant)
            yield assistant.role, reply

    def ask(
        self, content: Union[str, List[dict]], conversation: Optional[List[dict]] = None
    ) -> Union[str, List[tuple[str, str]]]:
        if conversation is None:
            conversation = []
        conversation.append({"role": "user", "content": content})

        results: List[tuple[str, str]] = []
        for assistant in self.assistants:
            reply = self._query_assistant(conversation, assistant)
            results.append((assistant.role, reply))

        if len(results) == 1:
            return results[0][1]

        return results


class MidjourneyBot(BaseBot):
    """Placeholder bot for Midjourney integration."""

    def ask_stream(
        self, content: Union[str, List[dict]], conversation: Optional[List[dict]] = None
    ) -> Iterable[tuple[str, str]]:
        yield "midjourney", "[Midjourney integration not implemented]"

    def ask(
        self, content: Union[str, List[dict]], conversation: Optional[List[dict]] = None
    ) -> Union[str, List[tuple[str, str]]]:
        return "[Midjourney integration not implemented]"


class ClaudeBot(BaseBot):
    """Placeholder bot for Claude.ai integration."""

    def ask_stream(
        self, content: Union[str, List[dict]], conversation: Optional[List[dict]] = None
    ) -> Iterable[tuple[str, str]]:
        yield "claude", "[Claude.ai integration not implemented]"

    def ask(
        self, content: Union[str, List[dict]], conversation: Optional[List[dict]] = None
    ) -> Union[str, List[tuple[str, str]]]:
        return "[Claude.ai integration not implemented]"


BOT_TYPES: dict[str, Type[BaseBot]] = {
    "openai": OpenAIBot,
    "midjourney": MidjourneyBot,
    "claude": ClaudeBot,
}


class TelegramBot:
    def __init__(self, bot_name: str | None = None) -> None:
        """Initialize the bot and Telegram application.

        The real Telegram objects are only constructed if the library is
        available. When running in the test environment, ``telegram`` is not
        installed and the fallbacks defined above use :class:`types.SimpleNamespace`.
        In that case we create minimal stub objects so that tests can run
        without raising ``AttributeError``.
        """
        if ApplicationBuilder is SimpleNamespace:
            # telegram library not installed - create stubs for tests
            self.application = SimpleNamespace(
                add_handler=lambda *a, **k: None,
                create_task=lambda coro: asyncio.create_task(coro),
                run_polling=lambda: None,
            )
        else:
            self.application = (
                ApplicationBuilder().token(settings.TELEGRAM_TOKEN).build()
            )
            self.application.add_handler(CommandHandler("start", self.start))
            self.application.add_handler(CommandHandler("clear", self.clear))
            self.application.add_handler(
                CommandHandler("consilium", self.start_consilium)
            )
            self.application.add_handler(
                CommandHandler("stopconsilium", self.stop_consilium)
            )
            self.application.add_handler(CommandHandler("models", self.list_models))
            self.application.add_handler(CommandHandler("model", self.set_model))
            self.application.add_handler(CommandHandler("bots", self.list_bots))
            self.application.add_handler(CommandHandler("bot", self.set_bot))
            self.application.add_handler(CallbackQueryHandler(self.handle_callback))
            self.application.add_handler(
                MessageHandler(filters.ALL, self.handle_message)
            )

        self.available_bots = list(BOT_TYPES.keys())
        self.bot_name = bot_name or getattr(settings, "DEFAULT_BOT", "openai")
        if self.bot_name not in self.available_bots:
            self.bot_name = "openai"

        if self.bot_name == "openai":
            setup_openai()
            self.available_models = self._fetch_models()
        else:
            self.available_models = []

        bot_class = BOT_TYPES[self.bot_name]
        self.bot: BaseBot = bot_class(model=settings.MODEL)
        # Per-chat specialized bots, used for the consilium mode
        self.bots: dict[int, BaseBot] = {}

        self.conversations: dict[int, List[dict]] = {}
        # Keep track of incoming media groups so that albums can be processed as
        # a single conversation unit. The mapping is ``media_group_id`` ->
        # ``{"messages": [...], "task": asyncio.Task | None}``.
        self._media_groups: dict[str, dict] = {}

    def _fetch_models(self) -> List[str]:
        """Return available OpenAI models or the default model if lookup fails."""
        try:
            if hasattr(openai, "models"):
                resp = openai.models.list()
            else:
                resp = openai.Model.list()
            data = (
                resp.get("data", [])
                if isinstance(resp, dict)
                else getattr(resp, "data", [])
            )
            models = [
                m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
                for m in data
            ]
            return [m for m in models if m]
        except Exception as exc:  # pragma: no cover - network issues
            logger.exception("Failed to fetch models: %s", exc)
            return [settings.MODEL]

    async def _send_chunks(self, send_func, text: str) -> None:
        """Send long text in chunks respecting Telegram limits."""
        limit = getattr(settings, "TELEGRAM_MAX_CHARS", 4096)
        for i in range(0, len(text), limit):
            await send_func(text[i : i + limit])

    async def _read_document_text(self, document) -> tuple[str, List[str]]:
        """Download a Telegram document and extract text and images from it."""
        file = await document.get_file()
        data = bytes(await file.download_as_bytearray())
        name = document.file_name.lower()
        images: List[str] = []
        try:
            if name.endswith(".pdf"):
                text = extract_text(BytesIO(data))
            elif name.endswith(".docx"):
                result = mammoth.convert_to_html(
                    BytesIO(data), convert_image=mammoth.images.data_uri
                )
                html = result.value
                tree = lxml.html.fromstring(html)
                text = "\n".join(t.strip() for t in tree.xpath("//text()") if t.strip())
                images = tree.xpath("//img/@src")
            elif name.endswith(".doc"):
                text = _extract_doc_text(data)
            elif name.endswith(".xlsx"):
                wb = openpyxl.load_workbook(
                    BytesIO(data), read_only=True, data_only=True
                )
                rows = []
                for ws in wb.worksheets:
                    for row in ws.iter_rows(values_only=True):
                        rows.append(",".join("" if c is None else str(c) for c in row))
                text = "\n".join(rows)
            elif name.endswith(".xls"):
                book = xlrd.open_workbook(file_contents=data)
                rows = []
                for sheet in book.sheets():
                    for row_idx in range(sheet.nrows):
                        row = sheet.row_values(row_idx)
                        rows.append(",".join(str(c) for c in row))
                text = "\n".join(rows)
            else:
                text = data.decode("utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover - fallback if parsing fails
            logger.exception("Failed to extract document text: %s", exc)
            return "", []
        return text[: settings.DOC_MAX_CHARS], images

    def _main_menu(self):
        keyboard = [
            [InlineKeyboardButton("Start consilium", callback_data="start_consilium")],
            [InlineKeyboardButton("Stop consilium", callback_data="stop_consilium")],
            [InlineKeyboardButton("Select model", callback_data="choose_model")],
            [InlineKeyboardButton("Select bot", callback_data="choose_bot")],
        ]
        return InlineKeyboardMarkup(keyboard)

    def _model_menu(self):
        keyboard = [
            [InlineKeyboardButton(m, callback_data=f"set_model:{m}")]
            for m in self.available_models
        ]
        keyboard.append([InlineKeyboardButton("Back", callback_data="back_main")])
        return InlineKeyboardMarkup(keyboard)

    def _bot_menu(self):
        keyboard = [
            [InlineKeyboardButton(b, callback_data=f"set_bot:{b}")]
            for b in self.available_bots
        ]
        keyboard.append([InlineKeyboardButton("Back", callback_data="back_main")])
        return InlineKeyboardMarkup(keyboard)

    async def handle_callback(self, update: Update, context: CallbackContext) -> None:
        query = update.callback_query
        data = query.data
        await getattr(query, "answer", lambda *a, **k: None)()

        if data == "start_consilium":
            await self.start_consilium(update, context)
            await query.edit_message_reply_markup(reply_markup=self._main_menu())
        elif data == "stop_consilium":
            await self.stop_consilium(update, context)
            await query.edit_message_reply_markup(reply_markup=self._main_menu())
        elif data == "choose_model":
            await query.edit_message_text(
                "Select model:", reply_markup=self._model_menu()
            )
        elif data == "choose_bot":
            await query.edit_message_text("Select bot:", reply_markup=self._bot_menu())
        elif data.startswith("set_model:"):
            model = data.split(":", 1)[1]
            context.args = [model]
            await self.set_model(update, context)
            await query.edit_message_text(
                f"Model set to {model}", reply_markup=self._main_menu()
            )
        elif data.startswith("set_bot:"):
            bot_name = data.split(":", 1)[1]
            context.args = [bot_name]
            await self.set_bot(update, context)
            await query.edit_message_text(
                f"Bot set to {bot_name}", reply_markup=self._main_menu()
            )
        elif data == "back_main":
            await query.edit_message_text(
                "Choose an action:", reply_markup=self._main_menu()
            )

    async def start(self, update: Update, context: CallbackContext) -> None:
        await update.message.reply_text(
            "Hello! Choose an action:", reply_markup=self._main_menu()
        )

    async def clear(self, update: Update, context: CallbackContext) -> None:
        chat_id = update.effective_chat.id
        self.conversations.pop(chat_id, None)
        await update.message.reply_text("Context cleared.")

    async def list_models(self, update: Update, context: CallbackContext) -> None:
        """List available models and show the current selection."""
        await update.message.reply_text(
            "Available models: "
            + ", ".join(self.available_models)
            + f"\nCurrent model: {self.bot.model}"
        )

    async def set_model(self, update: Update, context: CallbackContext) -> None:
        """Set the model used for OpenAI requests."""
        if not context.args:
            await self.list_models(update, context)
            return
        model = context.args[0]
        if model not in self.available_models:
            msg = getattr(update, "message", None) or update.callback_query.message
            await msg.reply_text(f"Model {model} not available.")
            return
        self.bot.model = model
        for b in self.bots.values():
            b.model = model
        msg = getattr(update, "message", None) or update.callback_query.message
        await msg.reply_text(f"Model set to {model}.")

    async def list_bots(self, update: Update, context: CallbackContext) -> None:
        """List available bot backends and show the current selection."""
        await update.message.reply_text(
            "Available bots: "
            + ", ".join(self.available_bots)
            + f"\nCurrent bot: {self.bot_name}"
        )

    async def set_bot(self, update: Update, context: CallbackContext) -> None:
        if not context.args:
            await self.list_bots(update, context)
            return
        bot_name = context.args[0]
        if bot_name not in self.available_bots:
            msg = getattr(update, "message", None) or update.callback_query.message
            await msg.reply_text(f"Bot {bot_name} not available.")
            return
        self.bot_name = bot_name
        bot_class = BOT_TYPES[bot_name]
        self.bot = bot_class(model=self.bot.model)
        for chat_id in list(self.bots.keys()):
            self.bots[chat_id] = bot_class(
                settings.CONSILIUM_ASSISTANTS, model=self.bot.model
            )
        msg = getattr(update, "message", None) or update.callback_query.message
        await msg.reply_text(f"Bot set to {bot_name}.")

    async def start_consilium(self, update: Update, context: CallbackContext) -> None:
        """Enable medical consilium mode for the chat."""
        chat_id = update.effective_chat.id
        bot_class = type(self.bot)
        self.bots[chat_id] = bot_class(
            settings.CONSILIUM_ASSISTANTS, model=self.bot.model
        )
        msg = getattr(update, "message", None) or update.callback_query.message
        await msg.reply_text("Consilium started.")

    async def stop_consilium(self, update: Update, context: CallbackContext) -> None:
        """Disable medical consilium mode for the chat."""
        chat_id = update.effective_chat.id
        self.bots.pop(chat_id, None)
        msg = getattr(update, "message", None) or update.callback_query.message
        await msg.reply_text("Consilium stopped.")

    async def handle_message(self, update: Update, context: CallbackContext) -> None:
        message = update.message
        text = message.caption or message.text or ""

        chat_id = update.effective_chat.id
        conversation = self.conversations.setdefault(chat_id, [])

        # Default content is just the text string
        content: Union[str, List[dict]] = text

        # Handle attached files
        if message.document:
            if message.media_group_id:
                group = self._media_groups.setdefault(
                    message.media_group_id, {"messages": [], "task": None}
                )
                group["messages"].append(message)
                if group["task"]:
                    group["task"].cancel()
                group["task"] = context.application.create_task(
                    self._process_media_group(message.media_group_id, context)
                )
                return
            file_name = message.document.file_name
            doc_text, doc_images = await self._read_document_text(message.document)
            if doc_text:
                snippet = doc_text.strip()
                prefix = f"Content of {file_name}:\n"
                text = f"{text}\n{prefix}{snippet}" if text else prefix + snippet
            else:
                text = (
                    f"{text}\n[User attached file: {file_name}]"
                    if text
                    else f"[User attached file: {file_name}]"
                )
            content = text
            if doc_images:
                content = [{"type": "text", "text": text}] + [
                    {"type": "image_url", "image_url": {"url": img}}
                    for img in doc_images
                ]

        if message.photo:
            if message.media_group_id:
                group = self._media_groups.setdefault(
                    message.media_group_id, {"messages": [], "task": None}
                )
                group["messages"].append(message)
                if group["task"]:
                    group["task"].cancel()
                group["task"] = context.application.create_task(
                    self._process_media_group(message.media_group_id, context)
                )
                return
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

        bot = self.bots.get(chat_id, self.bot)
        for role, text_part in bot.ask_stream(content, conversation):
            if len(conversation) > settings.CONTEXT_WINDOW_MESSAGES:
                self.conversations[chat_id] = conversation[
                    -settings.CONTEXT_WINDOW_MESSAGES :
                ]
            prefix = f"{role}: " if len(bot.assistants) > 1 else ""
            await self._send_chunks(message.reply_text, prefix + text_part)

    async def _process_media_group(
        self, group_id: str, context: CallbackContext
    ) -> None:
        await asyncio.sleep(1)
        group = self._media_groups.pop(group_id, None)
        if not group:
            return
        messages = group["messages"]
        text = ""
        for msg in messages:
            if not text:
                text = msg.caption or msg.text or ""
        doc_files: List[str] = []
        content: List[dict] = []
        if text:
            content.append({"type": "text", "text": text})
        for msg in messages:
            if msg.photo:
                photo = msg.photo[-1]
                file = await photo.get_file()
                image_bytes: Iterable[int] = await file.download_as_bytearray()
                b64 = base64.b64encode(bytearray(image_bytes)).decode("utf-8")
                image_url = f"data:image/jpeg;base64,{b64}"
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            if msg.document:
                doc_files.append(msg.document.file_name)
                doc_text, doc_images = await self._read_document_text(msg.document)
                if doc_text:
                    if content and content[0]["type"] == "text":
                        content[0][
                            "text"
                        ] += f"\nContent of {msg.document.file_name}:\n{doc_text.strip()}"
                    else:
                        content.insert(
                            0,
                            {
                                "type": "text",
                                "text": f"Content of {msg.document.file_name}:\n{doc_text.strip()}",
                            },
                        )
                for img in doc_images:
                    content.append({"type": "image_url", "image_url": {"url": img}})

        if doc_files:
            text_part = "User attached files: " + ", ".join(doc_files)
            if content and content[0]["type"] == "text":
                content[0]["text"] += "\n[" + text_part + "]"
            else:
                content.insert(0, {"type": "text", "text": "[" + text_part + "]"})

        bot = self.bots.get(messages[0].chat_id, self.bot)
        send_func = lambda t: context.bot.send_message(
            chat_id=messages[0].chat_id, text=t
        )
        for role, text_part in bot.ask_stream(
            content if len(content) > 1 else content[0]["text"] if content else ""
        ):
            prefix = f"{role}: " if len(bot.assistants) > 1 else ""
            await self._send_chunks(send_func, prefix + text_part)

    def run(self) -> None:
        logger.info("Starting bot")
        self.application.run_polling()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    TelegramBot().run()
