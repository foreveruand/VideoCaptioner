"""Tests for LLM client lifecycle behavior."""

from types import SimpleNamespace

from videocaptioner.core.llm import client as llm_client


class DummyOpenAI:
    """Simple OpenAI stub that records constructor args."""

    instances = []

    def __init__(self, base_url, api_key, http_client=None):
        self.base_url = base_url
        self.api_key = api_key
        self.http_client = http_client
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: kwargs)
        )
        DummyOpenAI.instances.append(self)


def test_get_llm_client_rebuilds_when_env_changes(monkeypatch):
    monkeypatch.setattr(llm_client, "OpenAI", DummyOpenAI)
    monkeypatch.setattr(llm_client, "create_logging_http_client", lambda: object())
    DummyOpenAI.instances = []
    llm_client.reset_llm_client()

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "key-a")
    client_a = llm_client.get_llm_client()

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "key-b")
    client_b = llm_client.get_llm_client()

    assert client_a is not client_b
    assert client_a.base_url == "https://api.openai.com/v1"
    assert client_b.base_url == "https://api.deepseek.com/v1"
    assert len(DummyOpenAI.instances) == 2
    llm_client.reset_llm_client()


def test_reset_llm_client_forces_rebuild(monkeypatch):
    monkeypatch.setattr(llm_client, "OpenAI", DummyOpenAI)
    monkeypatch.setattr(llm_client, "create_logging_http_client", lambda: object())
    DummyOpenAI.instances = []
    llm_client.reset_llm_client()

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "same-key")

    client_1 = llm_client.get_llm_client()
    llm_client.reset_llm_client()
    client_2 = llm_client.get_llm_client()

    assert client_1 is not client_2
    assert len(DummyOpenAI.instances) == 2
    llm_client.reset_llm_client()
