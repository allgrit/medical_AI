import types

import bot.settings as settings
from bot.bot import OpenAIBot, setup_openai


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
