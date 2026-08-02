import sys

import pytest
from PyQt5.QtWidgets import QApplication
from qfluentwidgets import ComboBox

from videocaptioner.core.entities import (
    LANGUAGES,
    BatchTaskType,
    TranscribeLanguageEnum,
    TranscribeModelEnum,
    get_available_transcribe_languages,
)
from videocaptioner.ui.common.config import cfg
from videocaptioner.ui.task_factory import TaskFactory
from videocaptioner.ui.view.batch_process_interface import BatchProcessInterface


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def transcribe_settings():
    items = [cfg.transcribe_model, cfg.transcribe_language, cfg.work_dir]
    snapshot = [(item, cfg.get(item)) for item in items]
    try:
        cfg.set(cfg.transcribe_model, TranscribeModelEnum.BIJIAN)
        cfg.set(cfg.transcribe_language, TranscribeLanguageEnum.CHINESE)
        yield
    finally:
        for item, value in snapshot:
            cfg.set(item, value)


def test_create_transcribe_task_uses_language_override(tmp_path, transcribe_settings):
    cfg.set(cfg.work_dir, str(tmp_path / "work-dir"))
    source = tmp_path / "sample.mp3"
    source.write_bytes(b"")

    default_task = TaskFactory.create_transcribe_task(str(source))
    task = TaskFactory.create_transcribe_task(
        str(source),
        need_next_task=True,
        transcribe_language=TranscribeLanguageEnum.ENGLISH,
    )

    assert default_task.transcribe_config is not None
    assert default_task.transcribe_config.transcribe_language == LANGUAGES["中文"]
    assert task.transcribe_config is not None
    assert task.transcribe_config.transcribe_language == LANGUAGES["英语"]
    assert task.output_path is not None
    assert task.output_path.endswith("-英语.srt")


def test_batch_task_uses_row_language_override(qapp, tmp_path, transcribe_settings, monkeypatch):
    source = tmp_path / "sample.mp3"
    source.write_bytes(b"")
    interface = BatchProcessInterface()
    interface.task_type_combo.setCurrentText(str(BatchTaskType.TRANSCRIBE))
    interface.add_files([str(source)])

    language_combo = interface.task_table.cellWidget(0, 1)
    assert isinstance(language_combo, ComboBox)
    assert language_combo.currentText() == "使用全局设置 (中文)"
    assert [language_combo.itemText(index) for index in range(language_combo.count())] == [
        "使用全局设置 (中文)",
        *[
            language.value
            for language in get_available_transcribe_languages(TranscribeModelEnum.BIJIAN)
        ],
    ]
    assert interface._get_row_transcribe_language(0) is None

    language_combo.setCurrentText(TranscribeLanguageEnum.ENGLISH.value)
    submitted_tasks = []
    monkeypatch.setattr(interface.batch_thread, "add_task", submitted_tasks.append)

    interface.start_task(str(source))

    assert len(submitted_tasks) == 1
    assert submitted_tasks[0].transcribe_language == TranscribeLanguageEnum.ENGLISH
    interface.close()


def test_batch_subtitle_task_has_no_source_language_selector(
    qapp, tmp_path, transcribe_settings
):
    source = tmp_path / "sample.srt"
    source.write_text("", encoding="utf-8")
    interface = BatchProcessInterface()
    interface.task_type_combo.setCurrentText(str(BatchTaskType.SUBTITLE))
    interface.add_files([str(source)])

    assert interface.task_table.cellWidget(0, 1) is None
    assert interface.task_table.item(0, 1).text() == "不适用"
    interface.close()
