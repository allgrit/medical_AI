"""Telegram bot that uses OpenAI models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Union, Iterable, Optional
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

try:  # Allow running tests without installed packages
    import openai
except Exception:  # pragma: no cover - handled in tests
    openai = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: None))
    )

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
            Assistant(role=a["role"], system_prompt=a["system_prompt"])
            for a in settings.ASSISTANTS
        ]
        setup_openai()

    def _query_assistant(self, conversation: List[dict], assistant: Assistant) -> str:
        messages = [
            {"role": "system", "content": assistant.system_prompt}
        ] + conversation
        logger.debug("Sending messages: %s", messages)
        resp = _create_chat_completion(model=settings.MODEL, messages=messages)
        if isinstance(resp, dict):
            content = resp["choices"][0]["message"]["content"]
        else:
            content = resp.choices[0].message.content
        conversation.append({"role": assistant.role, "content": content})
        return content

    def ask(
        self, content: Union[str, List[dict]], conversation: Optional[List[dict]] = None
    ) -> str:
        if conversation is None:
            conversation = []
        conversation.append({"role": "user", "content": content})
        reply = ""
        for assistant in self.assistants:
            reply = self._query_assistant(conversation, assistant)
        return reply


class TelegramBot:
    def __init__(self) -> None:
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
            self.application = ApplicationBuilder().token(settings.TELEGRAM_TOKEN).build()
            self.application.add_handler(CommandHandler("start", self.start))
            self.application.add_handler(CommandHandler("clear", self.clear))
            self.application.add_handler(CommandHandler("consilium", self.start_consilium))
            self.application.add_handler(CommandHandler("stopconsilium", self.stop_consilium))
            self.application.add_handler(MessageHandler(filters.ALL, self.handle_message))

        self.bot = OpenAIBot()
        # Per-chat specialized bots, used for the consilium mode
        self.bots: dict[int, OpenAIBot] = {}

        self.conversations: dict[int, List[dict]] = {}
        # Keep track of incoming media groups so that albums can be processed as
        # a single conversation unit. The mapping is ``media_group_id`` ->
        # ``{"messages": [...], "task": asyncio.Task | None}``.
        self._media_groups: dict[str, dict] = {}

    async def _read_document_text(self, document) -> tuple[str, List[str]]:
        """Download a Telegram document and extract text and images from it."""
        file = await document.get_file()
        data = bytes(await file.download_as_bytearray())
        name = document.file_name.lower()
        images: List[str] = []
        try:
            if name.endswith(".pdf"):
                text = extract_text(BytesIO(data))
            elif name.endswith(".docx") or name.endswith(".doc"):
                result = mammoth.convert_to_html(
                    BytesIO(data), convert_image=mammoth.images.data_uri
                )
                html = result.value
                tree = lxml.html.fromstring(html)
                text = "\n".join(
                    t.strip() for t in tree.xpath("//text()") if t.strip()
                )
                images = tree.xpath("//img/@src")
            elif name.endswith(".xlsx"):
                wb = openpyxl.load_workbook(BytesIO(data), read_only=True, data_only=True)
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

    async def start(self, update: Update, context: CallbackContext) -> None:
        await update.message.reply_text(
            "Hello! Send me a message and I will respond using OpenAI. "
            "Use /consilium to start a medical consilium and /stopconsilium to end it."
        )

    async def clear(self, update: Update, context: CallbackContext) -> None:
        chat_id = update.effective_chat.id
        self.conversations.pop(chat_id, None)
        await update.message.reply_text("Context cleared.")

    async def start_consilium(self, update: Update, context: CallbackContext) -> None:
        """Enable medical consilium mode for the chat."""
        chat_id = update.effective_chat.id
        self.bots[chat_id] = OpenAIBot(settings.CONSILIUM_ASSISTANTS)
        await update.message.reply_text("Consilium started.")

    async def stop_consilium(self, update: Update, context: CallbackContext) -> None:
        """Disable medical consilium mode for the chat."""
        chat_id = update.effective_chat.id
        self.bots.pop(chat_id, None)
        await update.message.reply_text("Consilium stopped.")

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
                text = f"{text}\n[User attached file: {file_name}]" if text else f"[User attached file: {file_name}]"
            content = text
            if doc_images:
                content = [
                    {"type": "text", "text": text}
                ] + [
                    {"type": "image_url", "image_url": {"url": img}}
                    for img in doc_images
                ]

        if message.photo:
            if message.media_group_id:
                group = self._media_groups.setdefault(message.media_group_id, {"messages": [], "task": None})
                group["messages"].append(message)
                if group["task"]:
                    group["task"].cancel()
                group["task"] = context.application.create_task(self._process_media_group(message.media_group_id, context))
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
        reply = bot.ask(content, conversation)
        if len(conversation) > settings.CONTEXT_WINDOW_MESSAGES:
            self.conversations[chat_id] = conversation[
                -settings.CONTEXT_WINDOW_MESSAGES :
            ]
        await message.reply_text(reply)

    async def _process_media_group(self, group_id: str, context: CallbackContext) -> None:
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
                        content[0]["text"] += f"\nContent of {msg.document.file_name}:\n{doc_text.strip()}"
                    else:
                        content.insert(0, {"type": "text", "text": f"Content of {msg.document.file_name}:\n{doc_text.strip()}"})
                for img in doc_images:
                    content.append({"type": "image_url", "image_url": {"url": img}})

        if doc_files:
            text_part = "User attached files: " + ", ".join(doc_files)
            if content and content[0]["type"] == "text":
                content[0]["text"] += "\n[" + text_part + "]"
            else:
                content.insert(0, {"type": "text", "text": "[" + text_part + "]"})

        bot = self.bots.get(messages[0].chat_id, self.bot)
        reply = bot.ask(content if len(content) > 1 else content[0]["text"] if content else "")
        await context.bot.send_message(chat_id=messages[0].chat_id, text=reply)

    def run(self) -> None:
        logger.info("Starting bot")
        self.application.run_polling()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    TelegramBot().run()
