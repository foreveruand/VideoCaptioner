import difflib
import json
import re
from dataclasses import dataclass
from typing import Any, List

from ..llm import call_llm
from ..llm.params import parse_llm_extra_params, prepare_llm_request_params
from ..prompts import get_prompt
from ..utils.logger import setup_logger
from ..utils.text_utils import count_words

logger = setup_logger("split_by_llm")

MAX_STEPS = 2  # Agent loop max retry count
LLM_SPLIT_SOFT_LIMIT_RATIO = 1.1
STRONG_BOUNDARY_PATTERN = r"[.!?。！？]"
SOFT_CJK_BOUNDARY_PATTERN = r"[，、；：。！？]"
SOFT_SPACE_BOUNDARY_PATTERN = r"[,:;.!?]"


@dataclass
class SplitValidationResult:
    status: str
    error_message: str
    content_preserved: bool


def split_by_llm(
    text: str,
    model: str = "gpt-4o-mini",
    max_word_count_cjk: int = 18,
    max_word_count_english: int = 12,
    llm_extra_params: Any = None,
    soft_limit_ratio: float = LLM_SPLIT_SOFT_LIMIT_RATIO,
    structured_output_mode: str = "off",
) -> List[str]:
    """使用LLM进行文本断句（固定使用句子Segments）

    Args:
        text: 待断句的文本
        model: LLM模型名称
        max_word_count_cjk: 中文最大字符数
        max_word_count_english: 英文最大单词数

    Returns:
        断句后的文本列表
    """
    return _split_with_agent_loop(
        text,
        model,
        max_word_count_cjk,
        max_word_count_english,
        llm_extra_params,
        soft_limit_ratio,
        structured_output_mode,
    )


def _split_with_agent_loop(
    text: str,
    model: str,
    max_word_count_cjk: int,
    max_word_count_english: int,
    llm_extra_params: Any = None,
    soft_limit_ratio: float = LLM_SPLIT_SOFT_LIMIT_RATIO,
    structured_output_mode: str = "off",
) -> List[str]:
    """使用agent loop 建立反馈循环进行文本断句，自动验证和修正"""
    prompt_path = "split/sentence"
    structured_output_mode = _normalize_structured_output_mode(structured_output_mode)
    extra_params = parse_llm_extra_params(llm_extra_params)
    response_format = _get_split_response_format(structured_output_mode)
    if response_format is None:
        response_format = extra_params.get("response_format")
    structured_output = isinstance(response_format, dict) and response_format.get("type") in {
        "json_object",
        "json_schema",
    }

    output_format = (
        'Return only a JSON object in the form {"segments": ["segment 1", "segment 2"]}. '
        "Do not include separators or any other fields."
        if structured_output
        else "Output the complete segmented text using <br> between segments only."
    )
    system_prompt = get_prompt(
        prompt_path,
        max_word_count_cjk=max_word_count_cjk,
        max_word_count_english=max_word_count_english,
        output_format=output_format,
        example_cjk_output=(
            '{"segments":["大家好","今天我们带来的3d创意设计作品是进制演示器","我是来自中山大学附属中学的方若涵"]}'
            if structured_output
            else "大家好<br>今天我们带来的3d创意设计作品是进制演示器<br>我是来自中山大学附属中学的方若涵"
        ),
        example_english_output=(
            '{"segments":["the upgraded claude sonnet is now available for all users","developers can build with the computer use beta"]}'
            if structured_output
            else "the upgraded claude sonnet is now available for all users<br>developers can build with the computer use beta"
        ),
    )

    user_prompt = (
        f"Split the following text without changing it. Hard limits: CJK segments must be "
        f"at most {max_word_count_cjk} characters; space-separated language segments must be "
        f"at most {max_word_count_english} words. {output_format}\n{text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    request_params = prepare_llm_request_params(
        {
            "temperature": 0.1,
            **({"response_format": response_format} if response_format else {}),
        },
        extra_params,
        protected_keys={"response_format"} if response_format else set(),
    )
    soft_limit_ratio = _normalize_soft_limit_ratio(soft_limit_ratio)

    for step in range(MAX_STEPS):
        response = call_llm(
            messages=messages,
            model=model,
            **request_params,
        )

        result_text = response.choices[0].message.content

        split_result = _parse_split_result(result_text, structured_output)
        split_result = _postprocess_split_result(
            split_result=split_result,
            original_text=text,
            max_word_count_cjk=max_word_count_cjk,
            max_word_count_english=max_word_count_english,
            soft_limit_ratio=soft_limit_ratio,
        )

        validation = _validate_split_result(
            original_text=text,
            split_result=split_result,
            max_word_count_cjk=max_word_count_cjk,
            max_word_count_english=max_word_count_english,
            soft_limit_ratio=soft_limit_ratio,
        )

        if validation.status == "valid":
            return split_result

        if step == MAX_STEPS - 1:
            break

        # 添加反馈到对话
        logger.warning(
            f"Split validation failed. Feedback loop (第{step + 1}次尝试):\n {validation.error_message}\n\n"
        )
        messages.append({"role": "assistant", "content": result_text})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Error: {validation.error_message}\n"
                    f"{output_format} Keep the original text unchanged and include ALL segments."
                ),
            }
        )

    raise RuntimeError(
        "LLM subtitle splitting did not produce a valid result after "
        f"{MAX_STEPS} attempts"
    )


def _normalize_soft_limit_ratio(value: float) -> float:
    if not isinstance(value, (int, float)) or value < 1:
        raise ValueError("LLM split soft limit ratio must be at least 1.0")
    return float(value)


def _normalize_structured_output_mode(value: str) -> str:
    if value not in {"off", "json_object", "json_schema"}:
        raise ValueError("Structured output mode must be off, json_object, or json_schema")
    return value


def _get_split_response_format(mode: str) -> dict[str, Any] | None:
    if mode == "off":
        return None
    if mode == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "subtitle_segments",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"segments": {"type": "array", "items": {"type": "string"}}},
                "required": ["segments"],
                "additionalProperties": False,
            },
        },
    }


def _parse_split_result(result_text: str, structured_output: bool = False) -> List[str]:
    if structured_output:
        try:
            payload = json.loads(result_text)
        except (TypeError, json.JSONDecodeError):
            return []
        segments = payload.get("segments") if isinstance(payload, dict) else None
        return segments if isinstance(segments, list) and all(isinstance(item, str) for item in segments) else []
    result_text_cleaned = result_text.replace("\n", "")
    return [segment.strip() for segment in result_text_cleaned.split("<br>")]


def _normalize_segment_spacing(segment: str, text_is_cjk: bool) -> str:
    if text_is_cjk:
        return re.sub(r"\s+", "", segment)
    return re.sub(r"\s+", " ", segment).strip()


def _is_cjk_text(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))


def _join_segments(split_result: List[str], text_is_cjk: bool) -> str:
    joiner = "" if text_is_cjk else " "
    return joiner.join(segment for segment in split_result if segment)


def _normalize_for_comparison(text: str, text_is_cjk: bool) -> str:
    if text_is_cjk or _is_cjk_text(text):
        return re.sub(r"\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _postprocess_split_result(
    split_result: List[str],
    original_text: str,
    max_word_count_cjk: int,
    max_word_count_english: int,
    soft_limit_ratio: float,
) -> List[str]:
    """对 LLM 结果做本地清洗与轻量修复。"""
    text_is_cjk = _is_cjk_text(original_text)
    max_allowed = max_word_count_cjk if text_is_cjk else max_word_count_english
    processed = [
        _normalize_segment_spacing(segment, text_is_cjk)
        for segment in split_result
        if segment and segment.strip()
    ]

    repaired: List[str] = []
    for segment in processed:
        repaired.extend(_split_overlong_segment_by_rules(segment, text_is_cjk, max_allowed, soft_limit_ratio))

    return _merge_tiny_segments_locally(repaired, text_is_cjk, max_allowed)


def _split_overlong_segment_by_rules(
    segment: str, text_is_cjk: bool, max_allowed: int, soft_limit_ratio: float
) -> List[str]:
    """仅对超长 LLM 段应用规则拆分，保留其余 LLM 结果。"""
    pending = [segment]
    repaired: List[str] = []
    while pending:
        current = pending.pop(0)
        parts = _split_overlong_segment_locally(
            current, text_is_cjk, max_allowed, soft_limit_ratio
        )
        if len(parts) == 1 and count_words(parts[0]) > max_allowed:
            parts = _force_split_overlong_segment(parts[0], text_is_cjk, max_allowed)
        if len(parts) == 1:
            repaired.extend(parts)
        else:
            pending = parts + pending
    return repaired


def _force_split_overlong_segment(
    segment: str, text_is_cjk: bool, max_allowed: int
) -> List[str]:
    if text_is_cjk:
        valid_indices = [
            index
            for index in range(1, len(segment))
            if count_words(segment[:index]) <= max_allowed
        ]
        if not valid_indices:
            return [segment]
        boundary_indices = [
            index for index in valid_indices if re.match(SOFT_CJK_BOUNDARY_PATTERN, segment[index - 1])
        ]
        split_index = boundary_indices[-1] if boundary_indices else valid_indices[-1]
        return [segment[:split_index].strip(), segment[split_index:].strip()]

    words = segment.split()
    if len(words) <= max_allowed:
        return [segment]
    return [" ".join(words[:max_allowed]), " ".join(words[max_allowed:])]


def _split_overlong_segment_locally(
    segment: str,
    text_is_cjk: bool,
    max_allowed: int,
    soft_limit_ratio: float,
) -> List[str]:
    """对轻微超长段做本地拆分。"""
    normalized_segment = _normalize_segment_spacing(segment, text_is_cjk)
    segment_word_count = count_words(normalized_segment)
    if segment_word_count <= max_allowed:
        return [normalized_segment]

    soft_limit = max_allowed * soft_limit_ratio
    if segment_word_count > soft_limit:
        return [normalized_segment]

    if text_is_cjk:
        split_index = _find_cjk_split_index(normalized_segment, max_allowed)
        if split_index is None:
            return [normalized_segment]
        left = normalized_segment[:split_index].strip()
        right = normalized_segment[split_index:].strip()
    else:
        words = normalized_segment.split()
        split_index = _find_space_split_index(words, max_allowed)
        if split_index is None:
            return [normalized_segment]
        left = " ".join(words[:split_index]).strip()
        right = " ".join(words[split_index:]).strip()

    if not left or not right:
        return [normalized_segment]

    return [left, right]


def _find_cjk_split_index(segment: str, max_allowed: int) -> int | None:
    boundary_positions = [
        idx + 1
        for idx, char in enumerate(segment[:-1])
        if re.match(SOFT_CJK_BOUNDARY_PATTERN, char)
    ]
    if not boundary_positions:
        return None
    return min(boundary_positions, key=lambda idx: abs(idx - max_allowed))


def _find_space_split_index(words: List[str], max_allowed: int) -> int | None:
    if len(words) <= 1:
        return None
    boundary_indices = []
    current_count = 0
    for index, word in enumerate(words[:-1], start=1):
        current_count += 1
        if current_count >= max(1, max_allowed - 2):
            boundary_indices.append(index)
        if re.search(SOFT_SPACE_BOUNDARY_PATTERN, word):
            boundary_indices.append(index)

    if not boundary_indices:
        return None
    return min(boundary_indices, key=lambda idx: abs(idx - max_allowed))


def _merge_tiny_segments_locally(
    split_result: List[str], text_is_cjk: bool, max_allowed: int
) -> List[str]:
    """合并异常碎的小片段，避免生成大量单字/单词段。"""
    if not split_result:
        return []

    joiner = "" if text_is_cjk else " "
    tiny_threshold = max(1, max_allowed // 4)
    merged = [split_result[0]]

    for segment in split_result[1:]:
        current = merged[-1]
        combined = f"{current}{joiner}{segment}" if not text_is_cjk else f"{current}{segment}"
        if (
            count_words(current) <= tiny_threshold
            and count_words(segment) <= tiny_threshold
            and count_words(combined) <= max_allowed
            and not re.search(STRONG_BOUNDARY_PATTERN, current[-1:])
        ):
            merged[-1] = combined
        else:
            merged.append(segment)

    return merged


def _validate_split_result(
    original_text: str,
    split_result: List[str],
    max_word_count_cjk: int,
    max_word_count_english: int,
    soft_limit_ratio: float = LLM_SPLIT_SOFT_LIMIT_RATIO,
) -> SplitValidationResult:
    """验证断句结果: 内容一致性、Segments数量、长度限制

    Returns:
        SplitValidationResult:
            - valid: 结果可直接使用
            - fixable: 仅存在轻微结构/长度问题
            - invalid: 内容被改写、为空或严重超限
    """
    non_empty_segments = [segment for segment in split_result if segment and segment.strip()]
    if not non_empty_segments:
        return SplitValidationResult(
            status="invalid",
            error_message="Invalid empty output",
            content_preserved=False,
        )

    text_is_cjk = _is_cjk_text(original_text)
    original_cleaned = _normalize_for_comparison(original_text, text_is_cjk)
    merged_cleaned = _normalize_for_comparison(
        _join_segments(non_empty_segments, text_is_cjk), text_is_cjk
    )

    if original_cleaned != merged_cleaned:
        return SplitValidationResult(
            status="invalid",
            error_message=_build_content_diff_message(
                original_cleaned, merged_cleaned, text_is_cjk
            ),
            content_preserved=False,
        )

    violations = []
    soft_violations = []
    max_allowed = max_word_count_cjk if text_is_cjk else max_word_count_english
    soft_limit = max_allowed * soft_limit_ratio

    for i, segment in enumerate(non_empty_segments, 1):
        word_count = count_words(segment)

        if word_count > max_allowed:
            segment_preview = segment[:40] + "..." if len(segment) > 40 else segment
            target_list = soft_violations if word_count <= soft_limit else violations
            target_list.append(
                f"Segment {i} '{segment_preview}': {word_count} {'chars' if text_is_cjk else 'words'} > {max_allowed} limit"
            )

    if violations:
        error_msg = "Unfixed length violations:\n" + "\n".join(
            f"- {violation}" for violation in violations
        )
        return SplitValidationResult(
            status="invalid",
            error_message=error_msg,
            content_preserved=True,
        )

    if soft_violations:
        error_msg = "Length violations:\n" + "\n".join(
            f"- {violation}" for violation in soft_violations
        )
        return SplitValidationResult(
            status="fixable",
            error_message=error_msg,
            content_preserved=True,
        )

    return SplitValidationResult(
        status="valid",
        error_message="",
        content_preserved=True,
    )


def _build_content_diff_message(
    original_cleaned: str, merged_cleaned: str, text_is_cjk: bool
) -> str:
    matcher = difflib.SequenceMatcher(None, original_cleaned, merged_cleaned)
    differences = []
    context_size = 5 if text_is_cjk else 20

    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            continue
        before = original_cleaned[max(0, a0 - context_size) : a0]
        after = original_cleaned[a1 : a1 + context_size]
        original_part = original_cleaned[a0:a1]
        merged_part = merged_cleaned[b0:b1]

        if opcode == "replace":
            differences.append(
                f"...{before}[{original_part}]{after}... -> changed to [{merged_part}]"
            )
        elif opcode == "delete":
            differences.append(f"...{before}[{original_part}]{after}... -> deleted")
        elif opcode == "insert":
            differences.append(
                f"Wrongly inserted [{merged_part}] between '...{before}' and '{after}...'"
            )

    error_msg = "Content modified:\n" + "\n".join(
        f"- {difference}" for difference in differences[:5]
    )
    error_msg += "\nKeep original text unchanged, only insert <br> between words."
    return error_msg


if __name__ == "__main__":
    sample_text = "大家好我叫杨玉溪来自有着良好音乐氛围的福建厦门自记事起我眼中的世界就是朦胧的童话书是各色杂乱的线条电视机是颜色各异的雪花小伙伴是只听其声不便骑行的马赛克后来我才知道这是一种眼底黄斑疾病虽不至于失明但终身无法治愈"
    sentences = split_by_llm(sample_text)
    print(f"断句结果 ({len(sentences)} 段):")
    for i, seg in enumerate(sentences, 1):
        print(f"  {i}. {seg}")
