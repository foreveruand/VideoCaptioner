"""Subtitle preprocessing helpers shared by optimization and translation."""

import re
from typing import List

from videocaptioner.core.asr.asr_data import ASRData, ASRDataSeg

MAX_REPEATED_SUBTITLE_GAP_MS = 3000


def _normalize_repeated_subtitle_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def merge_repeated_subtitle_segments(
    segments: List[ASRDataSeg],
    max_gap_ms: int = MAX_REPEATED_SUBTITLE_GAP_MS,
) -> List[ASRDataSeg]:
    """Merge adjacent identical source subtitles before LLM processing."""
    merged_segments: List[ASRDataSeg] = []
    i = 0

    while i < len(segments):
        seg = segments[i]
        normalized = _normalize_repeated_subtitle_text(seg.text)

        run_end = i + 1
        while run_end < len(segments):
            next_normalized = _normalize_repeated_subtitle_text(segments[run_end].text)
            if not normalized or next_normalized != normalized:
                break
            run_end += 1

        run = segments[i:run_end]
        if len(run) >= 3:
            seg.end_time = max(item.end_time for item in run)
            merged_segments.append(seg)
            i = run_end
            continue

        if merged_segments:
            prev = merged_segments[-1]
            prev_normalized = _normalize_repeated_subtitle_text(prev.text)
            time_gap = seg.start_time - prev.end_time
            if (
                normalized
                and normalized == prev_normalized
                and time_gap <= max_gap_ms
            ):
                prev.end_time = max(prev.end_time, seg.end_time)
                i += 1
                continue

        merged_segments.append(seg)
        i += 1

    return merged_segments


def preprocess_subtitle_before_llm(asr_data: ASRData) -> ASRData:
    """Apply deterministic cleanup before optimization or translation LLM calls."""
    asr_data.segments = merge_repeated_subtitle_segments(asr_data.segments)
    return asr_data
