"""Lightweight SubtitleThread logic tests that do not start Qt."""

from videocaptioner.core.asr.asr_data import ASRData, ASRDataSeg
from videocaptioner.core.entities import SubtitleConfig, SubtitleTask
from videocaptioner.core.subtitle.preprocess import preprocess_subtitle_before_llm
from videocaptioner.ui.thread.subtitle_thread import SubtitleThread


def test_word_timestamp_without_llm_split_does_not_need_llm():
    """字词级字幕在关闭LLM智能断句时仍可规则断句，不需要LLM配置。"""
    config = SubtitleConfig(
        need_split=False,
        need_optimize=False,
        need_translate=False,
    )
    asr_data = ASRData(
        [
            ASRDataSeg(text="你", start_time=0, end_time=100),
            ASRDataSeg(text="好", start_time=100, end_time=200),
        ]
    )
    thread = SubtitleThread(SubtitleTask(subtitle_config=config))

    assert asr_data.is_word_timestamp()
    assert thread.need_llm(config, asr_data) is False


def test_preprocess_before_llm_updates_visible_subtitle_data_shape():
    """手动字幕处理可用同一预处理结果刷新 UI 的 update_all 数据。"""
    asr_data = ASRData(
        [
            ASRDataSeg("もう少し休憩してください", 0, 1000),
            ASRDataSeg("もう少し休憩してください", 2500, 3500),
            ASRDataSeg("もう少し休憩してください", 6000, 7000),
        ]
    )

    result = preprocess_subtitle_before_llm(asr_data)
    visible_data = result.to_json()

    assert len(visible_data) == 1
    assert visible_data["1"]["start_time"] == 0
    assert visible_data["1"]["end_time"] == 7000
    assert visible_data["1"]["original_subtitle"] == "もう少し休憩してください"


def test_preprocess_merges_long_repeated_subtitle_runs():
    """三条及以上连续重复字幕即使间隔较长也应压缩为一条。"""
    asr_data = ASRData(
        [
            ASRDataSeg("そして戻ってきます...", 136184, 136434),
            ASRDataSeg("そして戻ってきます...", 136470, 136534),
            ASRDataSeg("そして戻ってきます。", 137052, 137124),
            ASRDataSeg("そして戻ってきます...", 137192, 137227),
            ASRDataSeg("次の字幕です", 138000, 139000),
        ]
    )

    result = preprocess_subtitle_before_llm(asr_data)

    assert len(result.segments) == 2
    assert result.segments[0].text == "そして戻ってきます..."
    assert result.segments[0].start_time == 136184
    assert result.segments[0].end_time == 137227
