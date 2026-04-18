"""LLM Translator integration tests.

Requires environment variables:
    OPENAI_BASE_URL: OpenAI-compatible API endpoint
    OPENAI_API_KEY: API key for authentication
    OPENAI_MODEL: Model name (optional, defaults to gpt-4o-mini)
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List

import pytest
from diskcache import Cache

from videocaptioner.core.asr.asr_data import ASRData, ASRDataSeg
from videocaptioner.core.llm.params import prepare_llm_request_params
from videocaptioner.core.subtitle.preprocess import preprocess_subtitle_before_llm
from videocaptioner.core.translate import SubtitleProcessData, TargetLanguage
from videocaptioner.core.translate.llm_translator import LLMTranslator
from videocaptioner.core.utils import cache


def _make_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.fixture
def mock_llm_client(monkeypatch):
    """Patch LLM calls with deterministic JSON responses."""

    def fake_stream_text(messages, model, on_text_delta=None, **kwargs):
        subtitle_dict = json.loads(messages[-1]["content"])
        response = json.dumps(
            {key: f"{value} translated" for key, value in subtitle_dict.items()},
            ensure_ascii=False,
        )
        if on_text_delta:
            on_text_delta(response, response)
        return response

    def fake_call_llm(messages, model, **kwargs):
        return _make_response(fake_stream_text(messages, model, **kwargs))

    monkeypatch.setattr(
        "videocaptioner.core.translate.llm_translator.call_llm_stream_text",
        fake_stream_text,
    )
    monkeypatch.setattr(
        "videocaptioner.core.translate.llm_translator.call_llm",
        fake_call_llm,
    )
    return fake_stream_text


@pytest.fixture
def isolated_llm_translator(tmp_path: Path, target_language: TargetLanguage):
    """Create a translator with isolated disk caches for item-cache tests."""
    cache.enable_cache()
    translator = LLMTranslator(
        thread_num=1,
        batch_num=5,
        target_language=target_language,
        model="gpt-4o-mini",
        custom_prompt="",
        is_reflect=False,
        update_callback=None,
    )
    translator._cache = Cache(str(tmp_path / "chunk_cache"))
    translator._item_cache = Cache(str(tmp_path / "item_cache"))
    yield translator
    translator._cache.close()
    translator._item_cache.close()
    cache.disable_cache()


@pytest.mark.integration
class TestLLMTranslator:
    """Test suite for LLMTranslator with OpenAI-compatible APIs."""

    @pytest.fixture
    def llm_translator(
        self, mock_llm_client, target_language: TargetLanguage
    ) -> LLMTranslator:
        """Create LLMTranslator instance for testing (using mock LLM)."""
        model = "gpt-4o-mini"

        return LLMTranslator(
            thread_num=2,
            batch_num=5,
            target_language=target_language,
            model=model,
            custom_prompt="",
            is_reflect=False,
            update_callback=None,
        )

    @pytest.mark.parametrize(
        "target_language",
        [TargetLanguage.SIMPLIFIED_CHINESE, TargetLanguage.JAPANESE],
    )
    def test_translate_simple_text(
        self,
        llm_translator: LLMTranslator,
        sample_asr_data: ASRData,
        expected_translations: Dict[str, Dict[str, List[str]]],
        target_language: TargetLanguage,
    ) -> None:
        """Test translating simple ASR data with quality validation (using mock LLM)."""

        result = llm_translator.translate_subtitle(sample_asr_data)

        print("\n" + "=" * 60)
        print(f"LLM Translation Results (to {target_language.value}):")
        for i, seg in enumerate(result.segments, 1):
            print(f"  [{i}] {seg.text} → {seg.translated_text}")
        print("=" * 60)

        assert len(result.segments) == len(sample_asr_data.segments)

        # Validate translation exists (quality check skipped for mock)
        for seg in result.segments:
            assert seg.translated_text, f"Translation is empty for: {seg.text}"

    def test_translate_chunk(
        self,
        llm_translator: LLMTranslator,
        sample_translate_data: list[SubtitleProcessData],
        expected_translations: Dict[str, Dict[str, List[str]]],
        target_language: TargetLanguage,
    ) -> None:
        """Test translating a single chunk of data with quality validation (using mock LLM)."""

        result = llm_translator._translate_chunk(sample_translate_data)

        print("\n" + "=" * 60)
        print(f"LLM Chunk Translation Results (to {target_language.value}):")
        for data in result:
            print(f"  [{data.index}] {data.original_text} → {data.translated_text}")
        print("=" * 60)

        assert len(result) == len(sample_translate_data)

        # Get expected keywords for target language
        expected_translations.get(target_language.value, {})

        # Validate translation exists (quality check skipped for mock)
        for data in result:
            assert (
                data.translated_text
            ), f"Translation is empty for: {data.original_text}"

    def test_structured_outputs_schema_and_parse(
        self,
        target_language: TargetLanguage,
    ) -> None:
        """Structured Outputs mode should request json_schema and parse wrapper output."""
        translator = LLMTranslator(
            thread_num=1,
            batch_num=5,
            target_language=target_language,
            model="gpt-4o-mini",
            custom_prompt="",
            is_reflect=False,
            use_structured_outputs=True,
            update_callback=None,
        )
        subtitle_dict = {"1": "hello", "2": "world"}

        kwargs = translator._get_response_format_kwargs(subtitle_dict)
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["response_format"]["json_schema"]["strict"] is True

        parsed = translator._parse_llm_response(
            '{"translations":[{"index":"1","translated_text":"你好"},'
            '{"index":"2","translated_text":"世界"}]}'
        )

        assert parsed == {"1": "你好", "2": "世界"}

    def test_llm_extra_params_are_stored(
        self,
        target_language: TargetLanguage,
    ) -> None:
        """JSON custom params should be parsed for LLM requests and cache keys."""
        translator = LLMTranslator(
            thread_num=1,
            batch_num=5,
            target_language=target_language,
            model="gpt-4o-mini",
            custom_prompt="",
            is_reflect=False,
            llm_extra_params='{"reasoning":{"effort":"high"}}',
            update_callback=None,
        )

        assert translator.llm_extra_params == {"reasoning": {"effort": "high"}}
        assert "reasoning" in translator._get_cache_key([])

        request_params = prepare_llm_request_params(
            {"temperature": 0.2},
            translator.llm_extra_params,
        )
        assert request_params["extra_body"] == {"reasoning": {"effort": "high"}}

    def test_repeated_source_segments_are_merged_before_llm_processing(
        self,
        target_language: TargetLanguage,
    ) -> None:
        """Repeated adjacent source subtitles should be merged before LLM processing."""
        asr_data = ASRData(
            [
                ASRDataSeg("もう少し休憩してください", 0, 1000),
                ASRDataSeg("もう少し休憩してください", 2500, 3500),
                ASRDataSeg("もう少し休憩してください", 6000, 7000),
            ]
        )

        result = preprocess_subtitle_before_llm(asr_data)

        assert len(result.segments) == 1
        assert result.segments[0].start_time == 0
        assert result.segments[0].end_time == 7000

    def test_item_cache_skips_cached_segments(
        self,
        isolated_llm_translator: LLMTranslator,
        sample_translate_data: list[SubtitleProcessData],
        monkeypatch,
    ) -> None:
        """Item-level cache should submit only uncached subtitles to the LLM."""
        translator = isolated_llm_translator
        translator._store_item_translation(
            sample_translate_data[0],
            "cached translation",
            status="complete",
        )
        submitted_payloads = []

        def fake_stream_text(messages, model, on_text_delta=None, **kwargs):
            payload = json.loads(messages[-1]["content"])
            submitted_payloads.append(payload)
            response = json.dumps(
                {key: f"translated {key}" for key in payload},
                ensure_ascii=False,
            )
            if on_text_delta:
                on_text_delta(response, response)
            return response

        monkeypatch.setattr(
            "videocaptioner.core.translate.llm_translator.call_llm_stream_text",
            fake_stream_text,
        )

        result = translator._translate_chunk(sample_translate_data)

        assert submitted_payloads == [
            {
                "2": "You are a teacher",
                "3": "VideoCaptioner is a tool for captioning videos",
            }
        ]
        assert result[0].translated_text == "cached translation"
        assert result[1].translated_text == "translated 2"
        assert result[2].translated_text == "translated 3"

    def test_item_cache_all_hits_skip_llm(
        self,
        isolated_llm_translator: LLMTranslator,
        sample_translate_data: list[SubtitleProcessData],
        monkeypatch,
    ) -> None:
        """When every subtitle has complete item cache, no LLM call is made."""
        translator = isolated_llm_translator
        for data in sample_translate_data:
            translator._store_item_translation(
                data,
                f"cached {data.index}",
                status="complete",
            )

        def fail_stream_text(*args, **kwargs):
            raise AssertionError("LLM should not be called")

        monkeypatch.setattr(
            "videocaptioner.core.translate.llm_translator.call_llm_stream_text",
            fail_stream_text,
        )

        result = translator._translate_chunk(sample_translate_data)

        assert [data.translated_text for data in result] == [
            "cached 1",
            "cached 2",
            "cached 3",
        ]

    def test_streaming_complete_json_writes_item_cache(
        self,
        isolated_llm_translator: LLMTranslator,
        sample_translate_data: list[SubtitleProcessData],
        monkeypatch,
    ) -> None:
        """Complete JSON items should be cached while streaming."""
        translator = isolated_llm_translator

        def fake_stream_text(messages, model, on_text_delta=None, **kwargs):
            chunks = ['{"1":"学生"', ',"2":"老师"', ',"3":"工具"}']
            full_content = ""
            for chunk in chunks:
                full_content += chunk
                if on_text_delta:
                    on_text_delta(chunk, full_content)
            return full_content

        monkeypatch.setattr(
            "videocaptioner.core.translate.llm_translator.call_llm_stream_text",
            fake_stream_text,
        )

        result = translator._translate_chunk(sample_translate_data)

        assert [data.translated_text for data in result] == ["学生", "老师", "工具"]
        assert translator._get_cached_item_translation(sample_translate_data[0]) == "学生"
        assert translator._get_cached_item_translation(sample_translate_data[1]) == "老师"
        assert translator._get_cached_item_translation(sample_translate_data[2]) == "工具"

    def test_streaming_interrupt_caches_completed_items_for_retry(
        self,
        isolated_llm_translator: LLMTranslator,
        sample_translate_data: list[SubtitleProcessData],
        monkeypatch,
    ) -> None:
        """Interrupted streams should keep completed item cache for the next run."""
        translator = isolated_llm_translator
        call_payloads = []

        def interrupted_stream_text(messages, model, on_text_delta=None, **kwargs):
            payload = json.loads(messages[-1]["content"])
            call_payloads.append(payload)
            full_content = '{"1":"学生",'
            if on_text_delta:
                on_text_delta(full_content, full_content)
            raise RuntimeError("stream interrupted")

        monkeypatch.setattr(
            "videocaptioner.core.translate.llm_translator.call_llm_stream_text",
            interrupted_stream_text,
        )

        with pytest.raises(RuntimeError, match="stream interrupted"):
            translator._translate_chunk(sample_translate_data[:2])

        assert translator._get_cached_item_translation(sample_translate_data[0]) == "学生"

        def retry_stream_text(messages, model, on_text_delta=None, **kwargs):
            payload = json.loads(messages[-1]["content"])
            call_payloads.append(payload)
            response = '{"2":"老师"}'
            if on_text_delta:
                on_text_delta(response, response)
            return response

        monkeypatch.setattr(
            "videocaptioner.core.translate.llm_translator.call_llm_stream_text",
            retry_stream_text,
        )

        result = translator._translate_chunk(sample_translate_data[:2])

        assert call_payloads == [
            {"1": "I am a student", "2": "You are a teacher"},
            {"2": "You are a teacher"},
        ]
        assert [data.translated_text for data in result] == ["学生", "老师"]

    def test_structured_stream_items_are_extracted(
        self,
        target_language: TargetLanguage,
    ) -> None:
        """Structured Outputs stream text should expose complete translation items."""
        translator = LLMTranslator(
            thread_num=1,
            batch_num=5,
            target_language=target_language,
            model="gpt-4o-mini",
            custom_prompt="",
            is_reflect=False,
            use_structured_outputs=True,
            update_callback=None,
        )

        parsed = translator._extract_complete_response_items(
            '{"translations":[{"index":"1","translated_text":"你好"},'
            '{"index":"2","translated_text":"世界"}'
        )

        assert parsed == {"1": "你好", "2": "世界"}

    def test_reflect_item_cache_uses_native_translation(
        self,
        tmp_path: Path,
        target_language: TargetLanguage,
    ) -> None:
        """Reflect mode item cache should store and reuse native_translation."""
        cache.enable_cache()
        translator = LLMTranslator(
            thread_num=1,
            batch_num=5,
            target_language=target_language,
            model="gpt-4o-mini",
            custom_prompt="",
            is_reflect=True,
            update_callback=None,
        )
        translator._item_cache = Cache(str(tmp_path / "reflect_item_cache"))
        data = SubtitleProcessData(index=1, original_text="hello")

        translator._store_item_translation(
            data,
            {
                "initial_translation": "初译",
                "reflection": "反思",
                "native_translation": "自然译文",
            },
            status="complete",
        )

        assert translator._get_cached_item_translation(data) == "自然译文"
        translator._item_cache.close()
        cache.disable_cache()

    def test_structured_outputs_reflect_parse(
        self,
        target_language: TargetLanguage,
    ) -> None:
        """Reflect mode should preserve native_translation from structured output."""
        translator = LLMTranslator(
            thread_num=1,
            batch_num=5,
            target_language=target_language,
            model="gpt-4o-mini",
            custom_prompt="",
            is_reflect=True,
            use_structured_outputs=True,
            update_callback=None,
        )

        parsed = translator._parse_llm_response(
            '{"translations":[{"index":"1","initial_translation":"初译",'
            '"reflection":"反思","native_translation":"自然译文"}]}'
        )

        assert parsed["1"]["native_translation"] == "自然译文"

    def test_cache_works(
        self,
        llm_translator: LLMTranslator,
        sample_asr_data: ASRData,
    ) -> None:
        """Test that caching mechanism works correctly (using mock LLM)."""
        cache.enable_cache()

        result1 = llm_translator.translate_subtitle(sample_asr_data)
        result2 = llm_translator.translate_subtitle(sample_asr_data)

        print("\n" + "=" * 60)
        print("LLM Cache Test:")
        print(f"  First call:  {result1.segments[-1].translated_text}")
        print(f"  Second call: {result2.segments[-1].translated_text}")
        print(
            f"  Match: {result1.segments[0].translated_text == result2.segments[0].translated_text}"
        )
        print("=" * 60)

        for seg1, seg2 in zip(result1.segments, result2.segments):
            assert seg1.translated_text == seg2.translated_text

    @pytest.mark.parametrize(
        "target_language",
        [TargetLanguage.SIMPLIFIED_CHINESE],
    )
    def test_reflect_translation(
        self,
        sample_asr_data: ASRData,
        target_language: TargetLanguage,
        check_env_vars: Callable,
    ) -> None:
        """Test reflect translation mode with nested dict validation."""
        check_env_vars("OPENAI_BASE_URL", "OPENAI_API_KEY")

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        translator = LLMTranslator(
            thread_num=2,
            batch_num=5,
            target_language=target_language,
            model=model,
            custom_prompt="",
            is_reflect=True,
            update_callback=None,
        )

        result = translator.translate_subtitle(sample_asr_data)

        print("\n" + "=" * 60)
        print(f"Reflect Translation Results (to {target_language.value}):")
        for i, seg in enumerate(result.segments, 1):
            print(f"  [{i}] {seg.text}")
            print(f"      → {seg.translated_text}")
        print("=" * 60)

        assert len(result.segments) == len(sample_asr_data.segments)

        for seg in result.segments:
            assert seg.translated_text, f"Translation is empty for: {seg.text}"
            assert len(seg.translated_text) > 0, "Translated text should not be empty"
