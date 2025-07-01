import types
import asyncio

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
