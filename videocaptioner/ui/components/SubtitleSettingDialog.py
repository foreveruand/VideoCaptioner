from qfluentwidgets import (
    BodyLabel,
    MessageBoxBase,
    SwitchSettingCard,
)
from qfluentwidgets import FluentIcon as FIF

from videocaptioner.ui.common.config import cfg
from videocaptioner.ui.components.SpinBoxSettingCard import (
    DoubleSpinBoxSettingCard,
    SpinBoxSettingCard,
)


class SubtitleSettingDialog(MessageBoxBase):
    """字幕设置对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = BodyLabel(self.tr("字幕设置"), self)

        # 创建设置卡片
        self.split_card = SwitchSettingCard(
            FIF.ALIGNMENT,
            self.tr("字幕分割"),
            self.tr("字幕是否使用大语言模型进行智能断句"),
            cfg.need_split,
            self,
        )

        self.word_count_cjk_card = SpinBoxSettingCard(
            cfg.max_word_count_cjk,
            FIF.TILES,  # type: ignore
            self.tr("中文最大字数"),
            self.tr("单条字幕的最大字数 (对于中日韩等字符)"),
            minimum=8,
            maximum=50,
            parent=self,
        )

        self.word_count_english_card = SpinBoxSettingCard(
            cfg.max_word_count_english,
            FIF.TILES,  # type: ignore
            self.tr("英文最大单词数"),
            self.tr("单条字幕的最大单词数 (英文)"),
            minimum=8,
            maximum=50,
            parent=self,
        )

        self.llm_chunk_target_multiplier_card = SpinBoxSettingCard(
            cfg.llm_chunk_target_multiplier,
            FIF.ALIGNMENT,  # type: ignore
            self.tr("LLM 分块倍率"),
            self.tr("单次 LLM 断句请求块长度 = 单行限制 × 该倍率"),
            minimum=1,
            maximum=50,
            parent=self,
        )

        self.llm_split_soft_limit_ratio_card = DoubleSpinBoxSettingCard(
            cfg.llm_split_soft_limit_ratio,
            FIF.ZOOM,  # type: ignore
            self.tr("轻微软超限比例"),
            self.tr("LLM 结果轻微超限时允许本地修复的阈值倍率"),
            minimum=1.0,
            maximum=2.0,
            decimals=2,
            step=0.05,
            parent=self,
        )

        # 添加到布局
        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.split_card)
        self.viewLayout.addWidget(self.word_count_cjk_card)
        self.viewLayout.addWidget(self.word_count_english_card)
        self.viewLayout.addWidget(self.llm_chunk_target_multiplier_card)
        self.viewLayout.addWidget(self.llm_split_soft_limit_ratio_card)
        # 设置间距
        self.viewLayout.setSpacing(10)

        # 设置窗口标题和宽度
        self.setWindowTitle(self.tr("字幕设置"))
        self.widget.setMinimumWidth(380)

        # 只显示取消按钮
        self.yesButton.hide()
        self.cancelButton.setText(self.tr("关闭"))
