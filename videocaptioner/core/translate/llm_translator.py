"""LLM 翻译器（使用 OpenAI）"""

import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import json_repair
import openai

from videocaptioner.core.asr.asr_data import ASRData
from videocaptioner.core.llm import call_llm, call_llm_stream_text
from videocaptioner.core.llm.params import (
    parse_llm_extra_params,
    prepare_llm_request_params,
    serialize_llm_extra_params,
)
from videocaptioner.core.prompts import get_prompt
from videocaptioner.core.subtitle.preprocess import preprocess_subtitle_before_llm
from videocaptioner.core.translate.base import (
    BaseTranslator,
    SubtitleProcessData,
    logger,
)
from videocaptioner.core.translate.types import TargetLanguage
from videocaptioner.core.utils.cache import generate_cache_key, is_cache_enabled


class LLMTranslator(BaseTranslator):
    """LLM 翻译器（OpenAI兼容API）"""

    MAX_STEPS = 3
    ITEM_CACHE_EXPIRE = 86400 * 30

    def __init__(
        self,
        thread_num: int,
        batch_num: int,
        target_language: TargetLanguage,
        model: str,
        custom_prompt: str,
        is_reflect: bool,
        use_structured_outputs: bool = False,
        llm_extra_params: Any = None,
        update_callback: Optional[Callable] = None,
    ):
        super().__init__(
            thread_num=thread_num,
            batch_num=batch_num,
            target_language=target_language,
            update_callback=update_callback,
        )

        self.model = model
        self.custom_prompt = custom_prompt
        self.is_reflect = is_reflect
        self.use_structured_outputs = use_structured_outputs
        self.llm_extra_params = parse_llm_extra_params(llm_extra_params)
        self._item_cache = self._cache

    def _preprocess_before_translate(self, subtitle_data: ASRData) -> ASRData:
        return preprocess_subtitle_before_llm(subtitle_data)

    def _translate_chunk(
        self, subtitle_chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        """翻译字幕块"""
        logger.debug(f"[+]正在翻译字幕: {subtitle_chunk[0].index} - {subtitle_chunk[-1].index}")

        pending_chunk = []
        for data in subtitle_chunk:
            cached_text = self._get_cached_item_translation(data)
            if cached_text is None:
                pending_chunk.append(data)
            else:
                data.translated_text = cached_text

        if not pending_chunk:
            return subtitle_chunk

        subtitle_by_key = {str(data.index): data for data in pending_chunk}
        subtitle_dict = {key: data.original_text for key, data in subtitle_by_key.items()}

        # 获取提示词
        if self.is_reflect:
            prompt = get_prompt(
                "translate/reflect",
                target_language=self.target_language,
                custom_prompt=self.custom_prompt,
            )
        else:
            prompt = get_prompt(
                "translate/standard",
                target_language=self.target_language,
                custom_prompt=self.custom_prompt,
            )

        try:
            # 使用agent loop进行翻译，自动验证和修正
            result_dict = self._agent_loop(prompt, subtitle_dict, subtitle_by_key)

            # 处理反思翻译模式的结果
            if self.is_reflect and isinstance(result_dict, dict):
                processed_result = {
                    k: f"{v.get('native_translation', v) if isinstance(v, dict) else v}"
                    for k, v in result_dict.items()
                }
            else:
                processed_result = {k: f"{v}" for k, v in result_dict.items()}

            # 将结果填充回SubtitleProcessData
            for data in pending_chunk:
                data.translated_text = processed_result.get(str(data.index), data.original_text)
            return subtitle_chunk
        except openai.RateLimitError as e:
            logger.error(f"OpenAI Rate Limit Error: {str(e)}")
            raise
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI Authentication Error: {str(e)}")
            raise
        except openai.NotFoundError as e:
            logger.error(f"OpenAI NotFound Error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"LLM translation error: {e}")
            raise
            return self._translate_chunk_single(subtitle_chunk)

    def _agent_loop(
        self,
        system_prompt: str,
        subtitle_dict: Dict[str, str],
        subtitle_by_key: Dict[str, SubtitleProcessData],
    ) -> Dict[str, Any]:
        """Agent loop翻译字幕块"""
        pending_dict = dict(subtitle_dict)
        completed_result: Dict[str, Any] = {}
        last_response_dict = None
        previous_response_content = ""
        error_message_for_next = ""
        # llm 反馈循环
        for _ in range(self.MAX_STEPS):
            cached_result = self._pop_cached_pending_items(pending_dict, subtitle_by_key)
            completed_result.update(cached_result)
            if not pending_dict:
                return completed_result

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(pending_dict, ensure_ascii=False)},
            ]
            if error_message_for_next:
                messages.append(
                    {
                        "role": "assistant",
                        "content": previous_response_content,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"Error: {error_message_for_next}\n\nFix the errors above and output ONLY a valid JSON dictionary with ALL {len(pending_dict)} keys",
                    }
                )
            request_params = prepare_llm_request_params(
                self._get_response_format_kwargs(pending_dict),
                self.llm_extra_params,
                protected_keys={"response_format"},
            )
            response_content = self._call_llm_text(
                messages=messages,
                request_params=request_params,
                pending_dict=pending_dict,
                subtitle_by_key=subtitle_by_key,
            )
            previous_response_content = response_content
            response_dict = self._parse_llm_response(response_content)
            last_response_dict = response_dict
            self._store_complete_items(response_dict, pending_dict, subtitle_by_key)
            cached_result = self._pop_cached_pending_items(pending_dict, subtitle_by_key)
            completed_result.update(cached_result)
            if not pending_dict:
                return completed_result

            is_valid, error_message = self._validate_llm_response(response_dict, pending_dict)
            if is_valid:
                return {**completed_result, **response_dict}
            else:
                error_message_for_next = error_message
                logger.warning(f"LLM translation response incomplete: {error_message}")

        cached_result = self._pop_cached_pending_items(pending_dict, subtitle_by_key)
        completed_result.update(cached_result)
        if not pending_dict:
            return completed_result

        missing_keys = sorted(pending_dict.keys(), key=lambda x: int(x) if x.isdigit() else x)
        raise RuntimeError(
            f"LLM translation did not return complete results for keys {missing_keys}. "
            f"Last response: {last_response_dict}"
        )

    def _call_llm_text(
        self,
        messages: List[dict],
        request_params: Dict[str, Any],
        pending_dict: Dict[str, str],
        subtitle_by_key: Dict[str, SubtitleProcessData],
    ) -> str:
        completed_keys: set[str] = set()
        stream_started = False

        def handle_stream_delta(_delta: str, full_content: str) -> None:
            nonlocal stream_started
            stream_started = True
            complete_items = self._extract_complete_response_items(full_content)
            for key, raw_result in complete_items.items():
                if key in completed_keys or key not in pending_dict:
                    continue
                self._store_item_translation(
                    subtitle_by_key[key],
                    raw_result,
                    status="complete",
                )
                completed_keys.add(key)

        for key in pending_dict:
            self._store_item_translation(
                subtitle_by_key[key],
                "",
                status="partial",
            )

        try:
            return call_llm_stream_text(
                messages=messages,
                model=self.model,
                on_text_delta=handle_stream_delta,
                **request_params,
            ).strip()
        except Exception:
            if stream_started:
                raise

            response = call_llm(
                messages=messages,
                model=self.model,
                **request_params,
            )
            return response.choices[0].message.content.strip()

    def _get_response_format_kwargs(self, subtitle_dict: Dict[str, str]) -> Dict[str, Any]:
        """Return OpenAI-compatible Structured Outputs args when enabled."""
        if not self.use_structured_outputs:
            return {}

        item_properties: Dict[str, Any] = {
            "index": {
                "type": "string",
                "enum": list(subtitle_dict.keys()),
            },
        }
        required_fields = ["index"]

        if self.is_reflect:
            item_properties.update(
                {
                    "initial_translation": {"type": "string"},
                    "reflection": {"type": "string"},
                    "native_translation": {"type": "string"},
                }
            )
            required_fields.extend(["initial_translation", "reflection", "native_translation"])
        else:
            item_properties["translated_text"] = {"type": "string"}
            required_fields.append("translated_text")

        schema = {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": item_properties,
                        "required": required_fields,
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["translations"],
            "additionalProperties": False,
        }

        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "subtitle_translation",
                    "strict": True,
                    "schema": schema,
                },
            }
        }

    def _get_item_cache_key(self, data: SubtitleProcessData) -> str:
        cache_data = {
            "class": self.__class__.__name__,
            "type": "llm_item_translation",
            "original_text": data.original_text,
            "target_language": self.target_language.value,
            "model": self.model,
            "custom_prompt": self.custom_prompt,
            "is_reflect": self.is_reflect,
            "use_structured_outputs": self.use_structured_outputs,
            "llm_extra_params": serialize_llm_extra_params(self.llm_extra_params),
        }
        return f"LLMTranslator:item:{generate_cache_key(cache_data)}"

    def _get_cached_item_translation(self, data: SubtitleProcessData) -> Optional[str]:
        if not is_cache_enabled():
            return None

        try:
            cached = self._item_cache.get(self._get_item_cache_key(data), default=None)
        except Exception:
            return None

        if not isinstance(cached, dict) or cached.get("status") != "complete":
            return None
        if cached.get("original_subtitle") != data.original_text:
            return None

        translated_text = cached.get("translated_subtitle")
        if not isinstance(translated_text, str) or not translated_text.strip():
            return None

        return translated_text

    def _store_item_translation(
        self,
        data: SubtitleProcessData,
        raw_result: Any,
        status: str,
    ) -> None:
        if not is_cache_enabled():
            return

        translated_text = ""
        if status == "complete":
            translated_text = self._extract_translated_text(raw_result)
            if not translated_text:
                return

        cache_value = {
            "original_subtitle": data.original_text,
            "translated_subtitle": translated_text,
            "status": status,
            "raw_result": raw_result,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            self._item_cache.set(
                self._get_item_cache_key(data),
                cache_value,
                expire=self.ITEM_CACHE_EXPIRE,
            )
        except Exception as e:
            logger.warning(f"Failed to write LLM item translation cache: {e}")

    def _extract_translated_text(self, raw_result: Any) -> str:
        if self.is_reflect and isinstance(raw_result, dict):
            return f"{raw_result.get('native_translation', '')}".strip()
        return f"{raw_result}".strip()

    def _pop_cached_pending_items(
        self,
        pending_dict: Dict[str, str],
        subtitle_by_key: Dict[str, SubtitleProcessData],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key in list(pending_dict.keys()):
            data = subtitle_by_key[key]
            cached_text = self._get_cached_item_translation(data)
            if cached_text is None:
                continue
            if self.is_reflect:
                result[key] = {"native_translation": cached_text}
            else:
                result[key] = cached_text
            data.translated_text = cached_text
            del pending_dict[key]
        return result

    def _store_complete_items(
        self,
        response_dict: Any,
        pending_dict: Dict[str, str],
        subtitle_by_key: Dict[str, SubtitleProcessData],
    ) -> None:
        if not isinstance(response_dict, dict):
            return

        for key, raw_result in response_dict.items():
            if key not in pending_dict:
                continue
            self._store_item_translation(
                subtitle_by_key[key],
                raw_result,
                status="complete",
            )

    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """Parse legacy dict output or Structured Outputs wrapper."""
        parsed = json_repair.loads(response_content)
        if not self.use_structured_outputs:
            return parsed

        if not isinstance(parsed, dict) or "translations" not in parsed:
            return parsed

        translations = parsed["translations"]
        if not isinstance(translations, list):
            return parsed

        result: Dict[str, Any] = {}
        for item in translations:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if index is None:
                continue
            key = str(index)
            if self.is_reflect:
                result[key] = {
                    "initial_translation": item.get("initial_translation", ""),
                    "reflection": item.get("reflection", ""),
                    "native_translation": item.get("native_translation", ""),
                }
            else:
                result[key] = item.get("translated_text", "")

        return result

    def _extract_complete_response_items(self, response_content: str) -> Dict[str, Any]:
        if self.use_structured_outputs:
            return self._extract_complete_structured_items(response_content)
        return self._extract_complete_dict_items(response_content)

    def _extract_complete_dict_items(self, response_content: str) -> Dict[str, Any]:
        decoder = json.JSONDecoder()
        result: Dict[str, Any] = {}
        object_start = response_content.find("{")
        if object_start < 0:
            return result

        i = object_start + 1
        text_len = len(response_content)
        while i < text_len:
            i = self._skip_json_separators(response_content, i)
            if i >= text_len or response_content[i] == "}":
                break
            if response_content[i] != '"':
                i += 1
                continue

            try:
                key, key_end = decoder.raw_decode(response_content, i)
            except ValueError:
                break
            if not isinstance(key, str):
                break

            i = self._skip_json_whitespace(response_content, key_end)
            if i >= text_len or response_content[i] != ":":
                break
            i = self._skip_json_whitespace(response_content, i + 1)

            try:
                value, value_end = decoder.raw_decode(response_content, i)
            except ValueError:
                break

            result[key] = value
            i = value_end

        return result

    def _extract_complete_structured_items(self, response_content: str) -> Dict[str, Any]:
        decoder = json.JSONDecoder()
        translations_pos = response_content.find('"translations"')
        if translations_pos < 0:
            return {}

        array_start = response_content.find("[", translations_pos)
        if array_start < 0:
            return {}

        result: Dict[str, Any] = {}
        i = array_start + 1
        text_len = len(response_content)
        while i < text_len:
            i = self._skip_json_separators(response_content, i)
            if i >= text_len or response_content[i] == "]":
                break

            try:
                item, item_end = decoder.raw_decode(response_content, i)
            except ValueError:
                break

            if isinstance(item, dict) and item.get("index") is not None:
                key = str(item["index"])
                if self.is_reflect:
                    result[key] = {
                        "initial_translation": item.get("initial_translation", ""),
                        "reflection": item.get("reflection", ""),
                        "native_translation": item.get("native_translation", ""),
                    }
                else:
                    result[key] = item.get("translated_text", "")
            i = item_end

        return result

    @staticmethod
    def _skip_json_whitespace(text: str, index: int) -> int:
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    @classmethod
    def _skip_json_separators(cls, text: str, index: int) -> int:
        while index < len(text):
            index = cls._skip_json_whitespace(text, index)
            if index < len(text) and text[index] == ",":
                index += 1
                continue
            break
        return index

    def _validate_llm_response(
        self, response_dict: Any, subtitle_dict: Dict[str, str]
    ) -> Tuple[bool, str]:
        """验证LLM翻译结果（支持普通和反思模式）

        Returns: (is_valid, error_feedback)
        """
        if not isinstance(response_dict, dict):
            return (
                False,
                f"Output must be a dict, got {type(response_dict).__name__}. Use format: {{'0': 'text', '1': 'text'}}",
            )

        expected_keys = set(subtitle_dict.keys())
        actual_keys = set(response_dict.keys())

        def sort_keys(keys):
            return sorted(keys, key=lambda x: int(x) if x.isdigit() else x)

        # 检查键是否匹配
        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            error_parts = []

            if missing:
                error_parts.append(
                    f"Missing keys {sort_keys(missing)} - you must translate these items"
                )
            if extra:
                error_parts.append(
                    f"Extra keys {sort_keys(extra)} - these keys are not in input, remove them"
                )

            return (False, "; ".join(error_parts))

        # 如果是反思模式，检查嵌套结构
        if self.is_reflect:
            for key, value in response_dict.items():
                if not isinstance(value, dict):
                    return (
                        False,
                        f"Key '{key}': value must be a dict with 'native_translation' field. Got {type(value).__name__}.",
                    )

                if "native_translation" not in value:
                    available_keys = list(value.keys())
                    return (
                        False,
                        f"Key '{key}': missing 'native_translation' field. Found keys: {available_keys}. Must include 'native_translation'.",
                    )

        return True, ""

    def _translate_chunk_single(
        self, subtitle_chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        """单条翻译模式"""
        single_prompt = get_prompt("translate/single", target_language=self.target_language)

        for data in subtitle_chunk:
            try:
                request_params = prepare_llm_request_params(
                    {"temperature": 0.7},
                    self.llm_extra_params,
                )
                response = call_llm(
                    messages=[
                        {"role": "system", "content": single_prompt},
                        {"role": "user", "content": data.original_text},
                    ],
                    model=self.model,
                    **request_params,
                )
                translated_text = response.choices[0].message.content.strip()
                data.translated_text = translated_text
            except Exception as e:
                logger.error(f"Single item translation failed {data.index}: {str(e)}")

        return subtitle_chunk

    def _get_cache_key(self, chunk: List[SubtitleProcessData]) -> str:
        """生成缓存键"""
        class_name = self.__class__.__name__
        chunk_key = generate_cache_key(chunk)
        lang = self.target_language.value
        model = self.model
        structured = self.use_structured_outputs
        extra_params = serialize_llm_extra_params(self.llm_extra_params)
        return f"{class_name}:{chunk_key}:{lang}:{model}:structured={structured}:extra={extra_params}"
