import os

from PyQt5.QtWidgets import QApplication
from qfluentwidgets import FluentIcon as FIF

from videocaptioner.ui.common.config import cfg
from videocaptioner.ui.components.LineEditSettingCard import LineEditSettingCard


def test_line_edit_preserves_cursor_when_config_is_written():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    original = cfg.get(cfg.openai_extra_params)
    card = LineEditSettingCard(cfg.openai_extra_params, FIF.CODE, "Test")
    try:
        card.lineEdit.setText("abcd")
        card.lineEdit.setCursorPosition(2)
        card.lineEdit.insert("X")
        app.processEvents()

        assert card.lineEdit.text() == "abXcd"
        assert card.lineEdit.cursorPosition() == 3

        cfg.set(cfg.openai_extra_params, "external")
        app.processEvents()
        assert card.lineEdit.text() == "external"
    finally:
        cfg.set(cfg.openai_extra_params, original)
        card.deleteLater()
