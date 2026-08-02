from pathlib import Path

from videocaptioner.ui.common.config import cfg
from videocaptioner.ui.task_factory import TaskFactory


def test_pipeline_work_dir_uses_source_stem(tmp_path):
    tracked_items = [
        cfg.work_dir,
        cfg.need_translate,
        cfg.transcribe_model,
        cfg.transcribe_language,
    ]
    original = [(item, cfg.get(item)) for item in tracked_items]
    try:
        cfg.set(cfg.work_dir, str(tmp_path / "work-dir"))
        cfg.set(cfg.need_translate, False)
        source = tmp_path / "demo-video.mp4"
        source.write_text("x", encoding="utf-8")

        transcribe_task = TaskFactory.create_transcribe_task(str(source), need_next_task=True)
        subtitle_task = TaskFactory.create_subtitle_task(
            transcribe_task.output_path or "", str(source), need_next_task=True
        )

        assert Path(transcribe_task.output_path or "").parts[-3] == source.stem
        assert Path(subtitle_task.output_path or "").parts[-3] == source.stem
    finally:
        for item, value in original:
            cfg.set(item, value)
