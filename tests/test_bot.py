import types
import asyncio
from io import BytesIO
import openpyxl
import docx
import textract

import bot.settings as settings
from bot.bot import OpenAIBot, setup_openai, TelegramBot


def test_setup_openai(monkeypatch):
    fake_openai = types.SimpleNamespace(api_key=None)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "key")
    monkeypatch.setattr("bot.bot.openai", fake_openai)
    setup_openai()
    assert fake_openai.api_key == "key"


def test_openai_bot_sequential(monkeypatch):
    assistants = [
        {"role": "assistant1", "system_prompt": "A"},
        {"role": "assistant2", "system_prompt": "B"},
    ]
    monkeypatch.setattr(settings, "ASSISTANTS", assistants)

    responses = [
        {"choices": [{"message": {"content": "first"}}]},
        {"choices": [{"message": {"content": "second"}}]},
    ]

    def fake_create(**kwargs):
        return responses.pop(0)

    monkeypatch.setattr("bot.bot._create_chat_completion", fake_create)
    bot_instance = OpenAIBot()
    reply = bot_instance.ask("hello")
    assert reply == "second"


def test_openai_bot_image(monkeypatch):
    content = [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]

    def fake_create(**kwargs):
        assert kwargs["messages"][1]["content"] == content
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr("bot.bot._create_chat_completion", fake_create)
    bot_instance = OpenAIBot()
    reply = bot_instance.ask(content)
    assert reply == "ok"



def test_openai_bot_conversation(monkeypatch):
    monkeypatch.setattr(
        settings, "ASSISTANTS", [{"role": "assistant", "system_prompt": "A"}]
    )

    replies = [
        {"choices": [{"message": {"content": "first"}}]},
        {"choices": [{"message": {"content": "second"}}]},
    ]

    def fake_create(**kwargs):
        return replies.pop(0)

    monkeypatch.setattr("bot.bot._create_chat_completion", fake_create)
    bot_instance = OpenAIBot()
    conv = []
    bot_instance.ask("hi", conv)
    bot_instance.ask("again", conv)

    assert conv == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "second"},
    ]

def test_telegram_bot_album(monkeypatch):
    async def run():
        responses = []

        class DummyPhoto:
            async def get_file(self):
                class F:
                    async def download_as_bytearray(self):
                        return b"img"

                return F()

        class DummyMessage:
            def __init__(self, caption=None):
                self.caption = caption
                self.text = None
                self.photo = [DummyPhoto()]
                self.document = None
                self.audio = None
                self.media_group_id = "g"
                self.chat_id = 1

        bot_instance = TelegramBot()
        monkeypatch.setattr(bot_instance.bot, "ask", lambda c: "album")

        async def send_message(chat_id, text):
            responses.append(text)

        ctx = types.SimpleNamespace(bot=types.SimpleNamespace(send_message=send_message))
        bot_instance._media_groups["g"] = {"messages": [DummyMessage("hi"), DummyMessage()], "task": None}
        await bot_instance._process_media_group("g", ctx)

        assert responses == ["album"]

    asyncio.run(run())


def test_telegram_bot_document_group(monkeypatch):
    async def run():
        responses = []

        class DummyDocument:
            file_name = "file.pdf"

        class DummyMessage:
            def __init__(self, caption=None):
                self.caption = caption
                self.text = None
                self.photo = None
                self.document = DummyDocument()
                self.audio = None
                self.media_group_id = "g"
                self.chat_id = 1

        bot_instance = TelegramBot()
        async def dummy_read(d):
            return "doc"
        monkeypatch.setattr(bot_instance, "_read_document_text", dummy_read)
        monkeypatch.setattr(bot_instance.bot, "ask", lambda c: "docs")

        async def send_message(chat_id, text):
            responses.append(text)

        ctx = types.SimpleNamespace(bot=types.SimpleNamespace(send_message=send_message))
        bot_instance._media_groups["g"] = {"messages": [DummyMessage("hi"), DummyMessage()], "task": None}
        await bot_instance._process_media_group("g", ctx)

        assert responses == ["docs"]

    asyncio.run(run())


def test_telegram_bot_single_document(monkeypatch):
    async def run():
        run.reply = None
        class DummyDocument:
            file_name = "report.pdf"

        class DummyMessage:
            caption = None
            text = None
            photo = None
            document = DummyDocument()
            audio = None
            media_group_id = None

            def __init__(self):
                self.chat_id = 1

            async def reply_text(self, text):
                run.reply = text

        bot_instance = TelegramBot()
        async def dummy_read(d):
            return "some text"
        monkeypatch.setattr(bot_instance, "_read_document_text", dummy_read)

        def fake_ask(content, conv):
            assert "some text" in content
            return "ok"

        monkeypatch.setattr(bot_instance.bot, "ask", fake_ask)
        update = types.SimpleNamespace(message=DummyMessage(), effective_chat=types.SimpleNamespace(id=1))
        ctx = types.SimpleNamespace(application=types.SimpleNamespace(create_task=lambda c: None))
        await bot_instance.handle_message(update, ctx)
        assert run.reply == "ok"

    asyncio.run(run())


def _dummy_document(name: str, data: bytes):
    class Dummy:
        file_name = name

        async def get_file(self):
            data_bytes = data

            class F:
                async def download_as_bytearray(self):
                    return data_bytes

            return F()

    return Dummy()


def test_read_document_text_csv():
    async def run():
        bot_instance = TelegramBot()
        doc = _dummy_document("data.csv", b"a,b\n1,2")
        text = await bot_instance._read_document_text(doc)
        assert "1,2" in text

    asyncio.run(run())


def test_read_document_text_xlsx():
    async def run():
        wb = openpyxl.Workbook()
        ws = wb.active
        ws["A1"] = "hello"
        bio = BytesIO()
        wb.save(bio)
        doc = _dummy_document("file.xlsx", bio.getvalue())
        bot_instance = TelegramBot()
        text = await bot_instance._read_document_text(doc)
        assert "hello" in text

    asyncio.run(run())


def test_read_document_text_docx():
    async def run():
        docx_file = docx.Document()
        docx_file.add_paragraph("test")
        bio = BytesIO()
        docx_file.save(bio)
        bot_instance = TelegramBot()
        doc = _dummy_document("doc.docx", bio.getvalue())
        text = await bot_instance._read_document_text(doc)
        assert "test" in text

    asyncio.run(run())


def test_read_document_text_doc(monkeypatch):
    async def run():
        bot_instance = TelegramBot()
        doc = _dummy_document("sample.doc", b"dummy")

        processed = {}

        def fake_process(filename):
            processed["file"] = filename
            return b"doc text"

        monkeypatch.setattr(textract, "process", fake_process)
        text = await bot_instance._read_document_text(doc)
        assert "doc text" in text
        assert processed

    asyncio.run(run())

