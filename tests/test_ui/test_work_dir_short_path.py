import json
from pathlib import Path

from videocaptioner.core.utils.work_dir_mapping import get_or_create_work_dir_short_name
from videocaptioner.ui.common.config import cfg
from videocaptioner.ui.task_factory import TaskFactory


def _snapshot(items):
    return [(item, cfg.get(item)) for item in items]


def _restore(snapshot):
    for item, value in snapshot:
        cfg.set(item, value)


def test_work_dir_short_name_mapping_persisted(tmp_path):
    work_dir = tmp_path / "work-dir"
    source = tmp_path / "very_long_name_video_file.mp4"
    source.write_text("x", encoding="utf-8")

    short_1 = get_or_create_work_dir_short_name(str(source), str(work_dir))
    short_2 = get_or_create_work_dir_short_name(str(source), str(work_dir))

    assert short_1 == short_2
    assert short_1.startswith("video_")

    map_file = work_dir / ".workdir_name_map.json"
    assert map_file.exists()
    data = json.loads(map_file.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert short_1 in data["short_to_meta"]
    assert data["short_to_meta"][short_1]["source_path"] == str(source)


def test_task_factory_uses_short_work_dir_paths_for_pipeline(tmp_path):
    tracked_items = [
        cfg.work_dir,
        cfg.need_translate,
        cfg.translator_service,
        cfg.transcribe_model,
        cfg.transcribe_language,
    ]
    snapshot = _snapshot(tracked_items)
    try:
        work_dir = tmp_path / "work-dir"
        cfg.set(cfg.work_dir, str(work_dir))
        cfg.set(cfg.need_translate, True)

        source = tmp_path / (
            "【耳舐め_ASMR】6月の花嫁_むちむちLカップ爆乳の白濁ウェディング_"
            "フェチに刺さるベトベト濃厚え〇ち＆耳舐め(初夜姿も_)【KU100】.mp4"
        )
        source.write_text("x", encoding="utf-8")

        transcribe_task = TaskFactory.create_transcribe_task(
            str(source), need_next_task=True
        )
        assert transcribe_task.output_path is not None
        transcribe_output = Path(transcribe_task.output_path)

        # work-dir 下不应再使用超长原始文件名作为目录名
        assert transcribe_output.parts[-3] != source.stem

        subtitle_task = TaskFactory.create_subtitle_task(
            file_path=str(transcribe_output),
            video_path=str(source),
            need_next_task=True,
        )
        assert subtitle_task.output_path is not None
        subtitle_output = Path(subtitle_task.output_path)

        # 转录与字幕阶段应落在同一短名目录中
        assert subtitle_output.parts[-3] == transcribe_output.parts[-3]
        assert subtitle_output.parts[-2] == "subtitle"
        assert str(subtitle_output).startswith(str(work_dir))
    finally:
        _restore(snapshot)
