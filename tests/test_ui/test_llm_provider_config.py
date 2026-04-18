"""Tests for provider-scoped LLM settings and migration."""

from videocaptioner.core.entities import LLMServiceEnum
from videocaptioner.ui.common.config import (
    cfg,
    migrate_legacy_llm_settings_if_needed,
    parse_llm_provider_presets,
)
from videocaptioner.ui.task_factory import TaskFactory


def _snapshot(items):
    return [(item, cfg.get(item)) for item in items]


def _restore(snapshot):
    for item, value in snapshot:
        cfg.set(item, value)


def test_task_factory_uses_provider_scoped_llm_settings():
    tracked_items = [
        cfg.llm_service,
        cfg.openai_extra_params,
        cfg.openai_use_structured_outputs,
        cfg.ollama_extra_params,
        cfg.ollama_use_structured_outputs,
    ]
    snapshot = _snapshot(tracked_items)
    try:
        cfg.set(cfg.llm_service, LLMServiceEnum.OPENAI)
        cfg.set(cfg.openai_extra_params, '{"reasoning":{"effort":"high"}}')
        cfg.set(cfg.openai_use_structured_outputs, True)
        cfg.set(cfg.ollama_extra_params, '{"temperature":0.3}')
        cfg.set(cfg.ollama_use_structured_outputs, False)

        openai_task = TaskFactory.create_subtitle_task("/tmp/demo.srt")
        assert openai_task.subtitle_config is not None
        assert (
            openai_task.subtitle_config.llm_extra_params
            == '{"reasoning":{"effort":"high"}}'
        )
        assert openai_task.subtitle_config.use_structured_outputs is True

        cfg.set(cfg.llm_service, LLMServiceEnum.OLLAMA)
        ollama_task = TaskFactory.create_subtitle_task("/tmp/demo.srt")
        assert ollama_task.subtitle_config is not None
        assert ollama_task.subtitle_config.llm_extra_params == '{"temperature":0.3}'
        assert ollama_task.subtitle_config.use_structured_outputs is False
    finally:
        _restore(snapshot)


def test_migrate_legacy_llm_settings_to_current_provider():
    tracked_items = [
        cfg.llm_service,
        cfg.openai_extra_params,
        cfg.openai_use_structured_outputs,
        cfg.llm_extra_params,
        cfg.use_structured_outputs,
    ]
    snapshot = _snapshot(tracked_items)
    try:
        cfg.set(cfg.llm_service, LLMServiceEnum.OPENAI)
        cfg.set(cfg.openai_extra_params, "")
        cfg.set(cfg.openai_use_structured_outputs, False)
        cfg.set(cfg.llm_extra_params, '{"reasoning":{"effort":"medium"}}')
        cfg.set(cfg.use_structured_outputs, True)

        migrate_legacy_llm_settings_if_needed()

        assert cfg.get(cfg.openai_extra_params) == '{"reasoning":{"effort":"medium"}}'
        assert cfg.get(cfg.openai_use_structured_outputs) is True
        assert cfg.get(cfg.llm_extra_params) == ""
        assert cfg.get(cfg.use_structured_outputs) is False
    finally:
        _restore(snapshot)


def test_parse_llm_provider_presets():
    raw = (
        '[{"name":"A","provider":"OpenAI 兼容"},'
        '{"name":"B","provider":"DeepSeek"},'
        '"invalid"]'
    )
    parsed = parse_llm_provider_presets(raw)
    assert len(parsed) == 2
    assert parsed[0]["name"] == "A"
    assert parse_llm_provider_presets("not-json") == []
