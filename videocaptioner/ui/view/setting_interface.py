import json
import webbrowser
from typing import Any

from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QFileDialog, QLabel, QWidget
from qfluentwidgets import (
    BodyLabel,
    ComboBoxSettingCard,
    CustomColorSettingCard,
    ExpandLayout,
    HyperlinkCard,
    InfoBar,
    LineEdit,
    MessageBoxBase,
    OptionsSettingCard,
    PrimaryPushSettingCard,
    PushSettingCard,
    RangeSettingCard,
    ScrollArea,
    SettingCardGroup,
    SwitchSettingCard,
    setTheme,
    setThemeColor,
)
from qfluentwidgets import FluentIcon as FIF

from videocaptioner.config import AUTHOR, FEEDBACK_URL, HELP_URL, RELEASE_URL, VERSION, YEAR
from videocaptioner.core.constant import (
    INFOBAR_DURATION_ERROR,
    INFOBAR_DURATION_SUCCESS,
    INFOBAR_DURATION_WARNING,
)
from videocaptioner.core.entities import LLMServiceEnum, TranscribeModelEnum, TranslatorServiceEnum
from videocaptioner.core.llm import check_whisper_connection
from videocaptioner.core.llm.check_llm import check_llm_connection, get_available_models
from videocaptioner.core.utils.cache import disable_cache, enable_cache
from videocaptioner.ui.common.config import cfg, get_provider_param_items, parse_llm_provider_presets
from videocaptioner.ui.common.signal_bus import signalBus
from videocaptioner.ui.components.EditComboBoxSettingCard import EditComboBoxSettingCard
from videocaptioner.ui.components.LineEditSettingCard import LineEditSettingCard


class SettingInterface(ScrollArea):
    """设置界面"""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(self.tr("设置"))
        self.scrollWidget = QWidget()
        self.expandLayout = ExpandLayout(self.scrollWidget)
        self.settingLabel = QLabel(self.tr("设置"), self)

        # 初始化所有设置组
        self.__initGroups()
        # 初始化所有配置卡片
        self.__initCards()
        # 初始化界面
        self.__initWidget()
        # 初始化布局
        self.__initLayout()
        # 连接信号和槽
        self.__connectSignalToSlot()

    def __initGroups(self):
        """初始化所有设置组"""
        # 转录配置组
        self.transcribeGroup = SettingCardGroup(self.tr("转录配置"), self.scrollWidget)
        # LLM配置组
        self.llmGroup = SettingCardGroup(self.tr("LLM配置"), self.scrollWidget)
        # 翻译服务组
        self.translate_serviceGroup = SettingCardGroup(
            self.tr("翻译服务"), self.scrollWidget
        )
        # 翻译与优化组
        self.translateGroup = SettingCardGroup(self.tr("翻译与优化"), self.scrollWidget)
        # 字幕合成配置组
        self.subtitleGroup = SettingCardGroup(
            self.tr("字幕合成配置"), self.scrollWidget
        )
        # 保存配置组
        self.saveGroup = SettingCardGroup(self.tr("保存配置"), self.scrollWidget)
        # 个性化组
        self.personalGroup = SettingCardGroup(self.tr("个性化"), self.scrollWidget)
        # 关于组
        self.aboutGroup = SettingCardGroup(self.tr("关于"), self.scrollWidget)

    def __initCards(self):
        """初始化所有配置卡片"""

        # ASR 服务配置卡片
        self.__createASRServiceCards()

        # LLM配置卡片
        self.__createLLMServiceCards()

        # 翻译配置卡片
        self.__createTranslateServiceCards()

        # 翻译与优化配置卡片
        self.subtitleCorrectCard = SwitchSettingCard(
            FIF.EDIT,
            self.tr("字幕校正"),
            self.tr("字幕处理过程是否对生成的字幕错别字、名词等进行校正"),
            cfg.need_optimize,
            self.translateGroup,
        )
        self.subtitleTranslateCard = SwitchSettingCard(
            FIF.LANGUAGE,
            self.tr("字幕翻译"),
            self.tr("字幕处理过程是否对生成的字幕进行翻译"),
            cfg.need_translate,
            self.translateGroup,
        )
        self.targetLanguageCard = ComboBoxSettingCard(
            cfg.target_language,
            FIF.LANGUAGE,
            self.tr("目标语言"),
            self.tr("选择翻译字幕的目标语言"),
            texts=[lang.value for lang in cfg.target_language.validator.options],  # type: ignore
            parent=self.translateGroup,
        )

        # 字幕合成配置卡片
        self.subtitleStyleCard = HyperlinkCard(
            "",
            self.tr("修改"),
            FIF.FONT,
            self.tr("字幕样式"),
            self.tr("选择字幕的样式（颜色、大小、字体等）"),
            self.subtitleGroup,
        )
        self.subtitleLayoutCard = HyperlinkCard(
            "",
            self.tr("修改"),
            FIF.FONT,
            self.tr("字幕布局"),
            self.tr("选择字幕的布局（单语、双语）"),
            self.subtitleGroup,
        )
        self.needVideoCard = SwitchSettingCard(
            FIF.VIDEO,
            self.tr("需要合成视频"),
            self.tr("开启时触发合成视频，关闭时跳过"),
            cfg.need_video,
            self.subtitleGroup,
        )
        self.softSubtitleCard = SwitchSettingCard(
            FIF.FONT,
            self.tr("软字幕"),
            self.tr("开启时字幕可在播放器中关闭或调整，关闭时字幕烧录到视频画面上"),
            cfg.soft_subtitle,
            self.subtitleGroup,
        )
        self.videoQualityCard = ComboBoxSettingCard(
            cfg.video_quality,
            FIF.SPEED_HIGH,
            self.tr("视频合成质量"),
            self.tr("硬字幕视频合成时的质量等级（质量越高文件越大，编码时间越长）"),
            texts=[quality.value for quality in cfg.video_quality.validator.options],  # type: ignore
            parent=self.subtitleGroup,
        )

        # 保存配置卡片
        self.savePathCard = PushSettingCard(
            self.tr("工作文件夹"),
            FIF.SAVE,
            self.tr("工作目录路径"),
            cfg.get(cfg.work_dir),
            self.saveGroup,
        )

        # 个性化配置卡片
        self.cacheEnabledCard = SwitchSettingCard(
            FIF.HISTORY,
            self.tr("启用缓存"),
            self.tr("相同配置下会复用之前的 ASR 和 LLM 结果；关闭缓存后每次重新生成"),
            cfg.cache_enabled,
            self.personalGroup,
        )
        self.themeCard = OptionsSettingCard(
            cfg.themeMode,
            FIF.BRUSH,
            self.tr("应用主题"),
            self.tr("更改应用程序的外观"),
            texts=[self.tr("浅色"), self.tr("深色"), self.tr("使用系统设置")],
            parent=self.personalGroup,
        )
        self.themeColorCard = CustomColorSettingCard(
            cfg.themeColor,
            FIF.PALETTE,
            self.tr("主题颜色"),
            self.tr("更改应用程序的主题颜色"),
            self.personalGroup,
        )
        self.zoomCard = OptionsSettingCard(
            cfg.dpiScale,
            FIF.ZOOM,
            self.tr("界面缩放"),
            self.tr("更改小部件和字体的大小"),
            texts=["100%", "125%", "150%", "175%", "200%", self.tr("使用系统设置")],
            parent=self.personalGroup,
        )
        self.languageCard = ComboBoxSettingCard(
            cfg.language,
            FIF.LANGUAGE,
            self.tr("语言"),
            self.tr("设置您偏好的界面语言"),
            texts=["简体中文", "繁體中文", "English", self.tr("使用系统设置")],
            parent=self.personalGroup,
        )

        # 关于卡片
        self.helpCard = HyperlinkCard(
            HELP_URL,
            self.tr("打开帮助页面"),
            FIF.HELP,
            self.tr("帮助"),
            self.tr("发现新功能并了解有关VideoCaptioner的使用技巧"),
            self.aboutGroup,
        )
        self.feedbackCard = PrimaryPushSettingCard(
            self.tr("提供反馈"),
            FIF.FEEDBACK,
            self.tr("提供反馈"),
            self.tr("提供反馈帮助我们改进VideoCaptioner"),
            self.aboutGroup,
        )
        self.aboutCard = PrimaryPushSettingCard(
            self.tr("检查更新"),
            FIF.INFO,
            self.tr("关于"),
            "© "
            + self.tr("版权所有")
            + f" {YEAR}, {AUTHOR}. "
            + self.tr("版本")
            + " "
            + VERSION,
            self.aboutGroup,
        )

        # 添加卡片到对应的组
        self.translateGroup.addSettingCard(self.subtitleCorrectCard)
        self.translateGroup.addSettingCard(self.subtitleTranslateCard)
        self.translateGroup.addSettingCard(self.targetLanguageCard)

        self.subtitleGroup.addSettingCard(self.subtitleStyleCard)
        self.subtitleGroup.addSettingCard(self.subtitleLayoutCard)
        self.subtitleGroup.addSettingCard(self.needVideoCard)
        self.subtitleGroup.addSettingCard(self.softSubtitleCard)
        self.subtitleGroup.addSettingCard(self.videoQualityCard)

        self.saveGroup.addSettingCard(self.savePathCard)
        self.saveGroup.addSettingCard(self.cacheEnabledCard)

        self.personalGroup.addSettingCard(self.themeCard)
        self.personalGroup.addSettingCard(self.themeColorCard)
        self.personalGroup.addSettingCard(self.zoomCard)
        self.personalGroup.addSettingCard(self.languageCard)

        self.aboutGroup.addSettingCard(self.helpCard)
        self.aboutGroup.addSettingCard(self.feedbackCard)
        self.aboutGroup.addSettingCard(self.aboutCard)

    def __createLLMServiceCards(self):
        """创建LLM服务相关的配置卡片"""
        self._updating_preset_ui = False

        # 服务选择卡片
        self.llmServiceCard = ComboBoxSettingCard(
            cfg.llm_service,
            FIF.ROBOT,
            self.tr("LLM 提供商"),
            self.tr("选择大模型提供商，用于字幕断句、优化、翻译"),
            texts=[service.value for service in cfg.llm_service.validator.options],  # type: ignore
            parent=self.llmGroup,
        )
        self.llmServiceCard.comboBox.setMinimumWidth(150)

        # 供应商预设
        self.llmPresetCard = ComboBoxSettingCard(
            cfg.llm_active_preset_name,
            FIF.SAVE,
            self.tr("LLM 预设"),
            self.tr("选择已保存的供应商配置预设"),
            texts=[],
            parent=self.llmGroup,
        )
        self.savePresetAsCard = PushSettingCard(
            self.tr("新建"),
            FIF.ADD,
            self.tr("保存当前为预设"),
            self.tr("保存当前供应商配置（包含 API Key）"),
            self.llmGroup,
        )
        self.overwritePresetCard = PushSettingCard(
            self.tr("覆盖"),
            FIF.SAVE,
            self.tr("覆盖当前预设"),
            self.tr("使用当前输入覆盖已选预设"),
            self.llmGroup,
        )
        self.deletePresetCard = PushSettingCard(
            self.tr("删除"),
            FIF.DELETE,
            self.tr("删除当前预设"),
            self.tr("删除已选预设配置"),
            self.llmGroup,
        )

        # 创建OPENAI官方API链接卡片
        self.openaiOfficialApiCard = HyperlinkCard(
            "https://api.videocaptioner.cn/register?aff=UrLB",
            self.tr("访问"),
            FIF.DEVELOPER_TOOLS,
            self.tr("VideoCaptioner 官方API"),
            self.tr("集成多种大语言模型，支持高并发字幕优化、翻译"),
            self.llmGroup,
        )
        # 默认隐藏
        self.openaiOfficialApiCard.setVisible(False)

        # 定义每个服务的配置
        service_configs = {
            LLMServiceEnum.OPENAI: {
                "prefix": "openai",
                "api_key_cfg": cfg.openai_api_key,
                "api_base_cfg": cfg.openai_api_base,
                "model_cfg": cfg.openai_model,
                "extra_params_cfg": cfg.openai_extra_params,
                "structured_outputs_cfg": cfg.openai_use_structured_outputs,
                "default_base": "https://api.openai.com/v1",
                "default_models": [
                    "gemini-2.5-pro",
                    "gpt-5",
                    "claude-sonnet-4-5-20250929",
                    "gemini-2.5-flash",
                    "claude-haiku-4-5-20251001",
                ],
            },
            LLMServiceEnum.SILICON_CLOUD: {
                "prefix": "silicon_cloud",
                "api_key_cfg": cfg.silicon_cloud_api_key,
                "api_base_cfg": cfg.silicon_cloud_api_base,
                "model_cfg": cfg.silicon_cloud_model,
                "extra_params_cfg": cfg.silicon_cloud_extra_params,
                "structured_outputs_cfg": cfg.silicon_cloud_use_structured_outputs,
                "default_base": "https://api.siliconflow.cn/v1",
                "default_models": [
                    "moonshotai/Kimi-K2-Instruct-0905",
                    "deepseek-ai/DeepSeek-V3",
                ],
            },
            LLMServiceEnum.DEEPSEEK: {
                "prefix": "deepseek",
                "api_key_cfg": cfg.deepseek_api_key,
                "api_base_cfg": cfg.deepseek_api_base,
                "model_cfg": cfg.deepseek_model,
                "extra_params_cfg": cfg.deepseek_extra_params,
                "structured_outputs_cfg": cfg.deepseek_use_structured_outputs,
                "default_base": "https://api.deepseek.com/v1",
                "default_models": ["deepseek-chat", "deepseek-reasoner"],
            },
            LLMServiceEnum.OLLAMA: {
                "prefix": "ollama",
                "api_key_cfg": cfg.ollama_api_key,
                "api_base_cfg": cfg.ollama_api_base,
                "model_cfg": cfg.ollama_model,
                "extra_params_cfg": cfg.ollama_extra_params,
                "structured_outputs_cfg": cfg.ollama_use_structured_outputs,
                "default_base": "http://localhost:11434/v1",
                "default_models": ["qwen3:8b"],
            },
            LLMServiceEnum.LM_STUDIO: {
                "prefix": "LM Studio",
                "api_key_cfg": cfg.lm_studio_api_key,
                "api_base_cfg": cfg.lm_studio_api_base,
                "model_cfg": cfg.lm_studio_model,
                "extra_params_cfg": cfg.lm_studio_extra_params,
                "structured_outputs_cfg": cfg.lm_studio_use_structured_outputs,
                "default_base": "http://localhost:1234/v1",
                "default_models": ["qwen3:8b"],
            },
            LLMServiceEnum.GEMINI: {
                "prefix": "gemini",
                "api_key_cfg": cfg.gemini_api_key,
                "api_base_cfg": cfg.gemini_api_base,
                "model_cfg": cfg.gemini_model,
                "extra_params_cfg": cfg.gemini_extra_params,
                "structured_outputs_cfg": cfg.gemini_use_structured_outputs,
                "default_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "default_models": [
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-2.0-flash-lite",
                ],
            },
            LLMServiceEnum.CHATGLM: {
                "prefix": "chatglm",
                "api_key_cfg": cfg.chatglm_api_key,
                "api_base_cfg": cfg.chatglm_api_base,
                "model_cfg": cfg.chatglm_model,
                "extra_params_cfg": cfg.chatglm_extra_params,
                "structured_outputs_cfg": cfg.chatglm_use_structured_outputs,
                "default_base": "https://open.bigmodel.cn/api/paas/v4",
                "default_models": ["glm-4-plus", "glm-4-air-250414", "glm-4-flash"],
            },
        }

        # 创建服务配置映射
        self.llm_service_configs = {}

        # 为每个服务创建配置卡片
        for service, config in service_configs.items():
            prefix = config["prefix"]

            # 创建API Key卡片
            api_key_card = LineEditSettingCard(
                config["api_key_cfg"],
                FIF.FINGERPRINT,
                self.tr("API Key"),
                self.tr(f"输入您的 {service.value} API Key"),
                "sk-" if service != LLMServiceEnum.OLLAMA else "",
                self.llmGroup,
            )
            setattr(self, f"{prefix}_api_key_card", api_key_card)

            # 创建Base URL卡片
            api_base_card = LineEditSettingCard(
                config["api_base_cfg"],
                FIF.LINK,
                self.tr("Base URL"),
                self.tr(f"输入 {service.value} Base URL"),
                config["default_base"],
                self.llmGroup,
            )
            setattr(self, f"{prefix}_api_base_card", api_base_card)

            # 设置只读状态：只有 OpenAI、Ollama、LM Studio 可以编辑 Base URL
            if service not in [
                LLMServiceEnum.OPENAI,
                LLMServiceEnum.OLLAMA,
                LLMServiceEnum.LM_STUDIO,
            ]:
                api_base_card.lineEdit.setReadOnly(True)

            # 创建模型选择卡片
            model_card = EditComboBoxSettingCard(
                config["model_cfg"],
                FIF.ROBOT,  # type: ignore
                self.tr("模型"),
                self.tr(f"选择 {service.value} 模型"),
                config["default_models"],
                self.llmGroup,
            )
            setattr(self, f"{prefix}_model_card", model_card)

            # 创建自定义参数卡片（供应商独立）
            extra_params_card = LineEditSettingCard(
                config["extra_params_cfg"],
                FIF.CODE,
                self.tr("LLM 自定义参数"),
                self.tr('JSON对象，例如 {"reasoning":{"effort":"high"}}'),
                '{"reasoning":{"effort":"high"}}',
                self.llmGroup,
            )
            setattr(self, f"{prefix}_extra_params_card", extra_params_card)

            # 创建结构化输出卡片（供应商独立）
            structured_outputs_card = SwitchSettingCard(
                FIF.CODE,
                self.tr("结构化输出"),
                self.tr("仅用于 LLM 翻译流程，需模型服务支持"),
                config["structured_outputs_cfg"],
                self.llmGroup,
            )
            setattr(self, f"{prefix}_structured_outputs_card", structured_outputs_card)

            # 存储服务配置
            cards = [
                api_key_card,
                api_base_card,
                model_card,
                extra_params_card,
                structured_outputs_card,
            ]

            self.llm_service_configs[service] = {
                "cards": cards,
                "api_base": api_base_card,
                "api_key": api_key_card,
                "model": model_card,
                "extra_params": extra_params_card,
                "structured_outputs": structured_outputs_card,
            }

        # 创建检查连接卡片
        self.checkLLMConnectionCard = PushSettingCard(
            self.tr("检查连接"),
            FIF.LINK,
            self.tr("检查 LLM 连接"),
            self.tr("点击检查 API 连接是否正常，并获取模型列表"),
            self.llmGroup,
        )

        self.__refreshLLMPresetItems()

        # 初始化显示状态
        self.__onLLMServiceChanged(self.llmServiceCard.comboBox.currentText())

    def __createASRServiceCards(self):
        """创建 Whisper API 配置卡片"""
        # 转录配置卡片
        self.transcribeModelCard = ComboBoxSettingCard(
            cfg.transcribe_model,
            FIF.MICROPHONE,
            self.tr("转录模型"),
            self.tr("语音转换文字要使用的语音识别服务"),
            texts=[model.value for model in cfg.transcribe_model.validator.options],  # type: ignore
            parent=self.transcribeGroup,
        )
        self.transcribeModelCard.comboBox.setMinimumWidth(150)

        # API Base URL
        self.whisperApiBaseCard = LineEditSettingCard(
            cfg.whisper_api_base,
            FIF.LINK,
            self.tr("Whisper API Base URL"),
            self.tr("输入 Whisper API Base URL"),
            "https://api.openai.com/v1",
            self.transcribeGroup,
        )

        # API Key
        self.whisperApiKeyCard = LineEditSettingCard(
            cfg.whisper_api_key,
            FIF.FINGERPRINT,
            self.tr("Whisper API Key"),
            self.tr("输入 Whisper API Key"),
            "sk-",
            self.transcribeGroup,
        )

        # 模型选择
        self.whisperApiModelCard = EditComboBoxSettingCard(
            cfg.whisper_api_model,
            FIF.ROBOT,  # type: ignore
            self.tr("Whisper 模型"),
            self.tr("选择 Whisper 模型"),
            [
                "whisper-1",
                "whisper-large-v3-turbo",
            ],
            self.transcribeGroup,
        )

        # 测试连接按钮
        self.checkWhisperConnectionCard = PushSettingCard(
            self.tr("测试 Whisper 连接"),
            FIF.CONNECT,
            self.tr("测试 Whisper API 连接"),
            self.tr("点击测试 API 连接是否正常"),
            self.transcribeGroup,
        )

        # 默认隐藏 Whisper API 配置卡片（仅在选择 Whisper API 时显示）
        self.whisperApiBaseCard.setVisible(False)
        self.whisperApiKeyCard.setVisible(False)
        self.whisperApiModelCard.setVisible(False)
        self.checkWhisperConnectionCard.setVisible(False)

    def __createTranslateServiceCards(self):
        """创建翻译服务相关的配置卡片"""
        # 翻译服务选择卡片
        self.translatorServiceCard = ComboBoxSettingCard(
            cfg.translator_service,
            FIF.ROBOT,
            self.tr("翻译服务"),
            self.tr("选择翻译服务"),
            texts=[
                service.value
                for service in cfg.translator_service.validator.options  # type: ignore
            ],
            parent=self.translate_serviceGroup,
        )
        self.translatorServiceCard.comboBox.setMinimumWidth(150)

        # 反思翻译开关
        self.needReflectTranslateCard = SwitchSettingCard(
            FIF.EDIT,
            self.tr("需要反思翻译"),
            self.tr("启用反思翻译可以提高翻译质量，但耗费更多时间和token"),
            cfg.need_reflect_translate,
            self.translate_serviceGroup,
        )

        # DeepLx端点配置
        self.deeplxEndpointCard = LineEditSettingCard(
            cfg.deeplx_endpoint,
            FIF.LINK,
            self.tr("DeepLx 后端"),
            self.tr("输入 DeepLx 的后端地址(开启deeplx翻译时必填)"),
            "https://api.deeplx.org/translate",
            self.translate_serviceGroup,
        )

        # 批处理大小配置
        self.batchSizeCard = RangeSettingCard(
            cfg.batch_size,
            FIF.ALIGNMENT,
            self.tr("批处理大小"),
            self.tr("每批处理字幕的数量，建议为 10 的倍数"),
            parent=self.translate_serviceGroup,
        )

        # 线程数配置
        self.threadNumCard = RangeSettingCard(
            cfg.thread_num,
            FIF.SPEED_HIGH,
            self.tr("线程数"),
            self.tr(
                "请求并行处理的数量，模型服务商允许的情况下建议尽可能大，数值越大速度越快"
            ),
            parent=self.translate_serviceGroup,
        )

        # 添加卡片到翻译服务组
        self.translate_serviceGroup.addSettingCard(self.translatorServiceCard)
        self.translate_serviceGroup.addSettingCard(self.needReflectTranslateCard)
        self.translate_serviceGroup.addSettingCard(self.deeplxEndpointCard)
        self.translate_serviceGroup.addSettingCard(self.batchSizeCard)
        self.translate_serviceGroup.addSettingCard(self.threadNumCard)

        # 初始化显示状态
        self.__onTranslatorServiceChanged(
            self.translatorServiceCard.comboBox.currentText()
        )

    def __initWidget(self):
        self.resize(1000, 800)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore
        self.setViewportMargins(0, 80, 0, 20)
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.setObjectName("settingInterface")

        # 初始化样式表
        self.scrollWidget.setObjectName("scrollWidget")
        self.settingLabel.setObjectName("settingLabel")

        # 初始化转录模型配置卡片的显示状态
        self.__onTranscribeModelChanged(self.transcribeModelCard.comboBox.currentText())

        # 初始化翻译服务配置卡片的显示状态
        self.__onTranslatorServiceChanged(
            self.translatorServiceCard.comboBox.currentText()
        )

        self.setStyleSheet(
            """
            SettingInterface, #scrollWidget {
                background-color: transparent;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QLabel#settingLabel {
                font: 33px 'Microsoft YaHei';
                background-color: transparent;
                color: white;
            }
        """
        )

    def __initLayout(self):
        """初始化布局"""
        self.settingLabel.move(36, 30)

        # 添加转录配置卡片
        self.transcribeGroup.addSettingCard(self.transcribeModelCard)
        # 添加 Whisper API 配置卡片
        self.transcribeGroup.addSettingCard(self.whisperApiBaseCard)
        self.transcribeGroup.addSettingCard(self.whisperApiKeyCard)
        self.transcribeGroup.addSettingCard(self.whisperApiModelCard)
        self.transcribeGroup.addSettingCard(self.checkWhisperConnectionCard)

        # 添加LLM配置卡片
        self.llmGroup.addSettingCard(self.llmServiceCard)
        self.llmGroup.addSettingCard(self.llmPresetCard)
        self.llmGroup.addSettingCard(self.savePresetAsCard)
        self.llmGroup.addSettingCard(self.overwritePresetCard)
        self.llmGroup.addSettingCard(self.deletePresetCard)
        # 添加OPENAI官方API链接卡片
        self.llmGroup.addSettingCard(self.openaiOfficialApiCard)
        for config in self.llm_service_configs.values():
            for card in config["cards"]:
                self.llmGroup.addSettingCard(card)
        self.llmGroup.addSettingCard(self.checkLLMConnectionCard)

        # 将所有组添加到布局
        self.expandLayout.setSpacing(28)
        self.expandLayout.setContentsMargins(36, 10, 36, 0)
        self.expandLayout.addWidget(self.transcribeGroup)
        self.expandLayout.addWidget(self.llmGroup)
        self.expandLayout.addWidget(self.translate_serviceGroup)
        self.expandLayout.addWidget(self.translateGroup)
        self.expandLayout.addWidget(self.subtitleGroup)
        self.expandLayout.addWidget(self.saveGroup)
        self.expandLayout.addWidget(self.personalGroup)
        self.expandLayout.addWidget(self.aboutGroup)

    def __connectSignalToSlot(self):
        """连接信号与槽"""
        cfg.appRestartSig.connect(self.__showRestartTooltip)

        # LLM服务切换
        self.llmServiceCard.comboBox.currentTextChanged.connect(
            self.__onLLMServiceChanged
        )
        self.llmPresetCard.comboBox.currentTextChanged.connect(self.__onLLMPresetChanged)
        self.savePresetAsCard.clicked.connect(self.__saveCurrentAsPreset)
        self.overwritePresetCard.clicked.connect(self.__overwriteCurrentPreset)
        self.deletePresetCard.clicked.connect(self.__deleteCurrentPreset)

        # 翻译服务切换
        self.translatorServiceCard.comboBox.currentTextChanged.connect(
            self.__onTranslatorServiceChanged
        )

        # 转录模型切换
        self.transcribeModelCard.comboBox.currentTextChanged.connect(
            self.__onTranscribeModelChanged
        )

        # 检查 LLM 连接
        self.checkLLMConnectionCard.clicked.connect(self.checkLLMConnection)

        # 检查 Whisper 连接
        self.checkWhisperConnectionCard.clicked.connect(self.checkWhisperConnection)

        # 保存路径
        self.savePathCard.clicked.connect(self.__onsavePathCardClicked)

        # 字幕样式修改跳转
        self.subtitleStyleCard.linkButton.clicked.connect(
            lambda: self.window().switchTo(self.window().subtitleStyleInterface)  # type: ignore
        )
        self.subtitleLayoutCard.linkButton.clicked.connect(
            lambda: self.window().switchTo(self.window().subtitleStyleInterface)  # type: ignore
        )

        # 个性化
        self.cacheEnabledCard.checkedChanged.connect(self.__onCacheEnabledChanged)
        self.themeCard.optionChanged.connect(lambda ci: setTheme(cfg.get(ci)))
        self.themeColorCard.colorChanged.connect(setThemeColor)

        # 反馈
        self.feedbackCard.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(FEEDBACK_URL))  # type: ignore
        )

        # 关于
        self.aboutCard.clicked.connect(self.checkUpdate)

        # 全局 signalBus
        self.transcribeModelCard.comboBox.currentTextChanged.connect(
            signalBus.transcription_model_changed
        )
        self.subtitleCorrectCard.checkedChanged.connect(
            signalBus.subtitle_optimization_changed
        )
        self.subtitleTranslateCard.checkedChanged.connect(
            signalBus.subtitle_translation_changed
        )
        self.targetLanguageCard.comboBox.currentTextChanged.connect(
            signalBus.target_language_changed
        )
        self.softSubtitleCard.checkedChanged.connect(signalBus.soft_subtitle_changed)
        self.needVideoCard.checkedChanged.connect(signalBus.need_video_changed)
        self.videoQualityCard.comboBox.currentTextChanged.connect(
            signalBus.video_quality_changed
        )

    def __showRestartTooltip(self):
        """显示重启提示"""
        InfoBar.success(
            self.tr("更新成功"),
            self.tr("配置将在重启后生效"),
            duration=INFOBAR_DURATION_SUCCESS,
            parent=self,
        )

    def __onsavePathCardClicked(self):
        """处理保存路径卡片点击事件"""
        folder = QFileDialog.getExistingDirectory(self, self.tr("选择文件夹"), "./")
        if not folder or cfg.get(cfg.work_dir) == folder:
            return
        cfg.set(cfg.work_dir, folder)
        self.savePathCard.setContent(folder)

    def __onCacheEnabledChanged(self, is_enabled: bool):
        """处理缓存开关变化"""
        if is_enabled:
            enable_cache()
            InfoBar.success(
                self.tr("缓存已启用"),
                self.tr("ASR、翻译等操作将优先使用缓存"),
                duration=INFOBAR_DURATION_SUCCESS,
                parent=self,
            )
        else:
            disable_cache()
            InfoBar.warning(
                self.tr("缓存已禁用"),
                self.tr("所有操作将重新生成，不使用缓存（建议开启缓存）"),
                duration=INFOBAR_DURATION_WARNING,
                parent=self,
            )

    def checkLLMConnection(self):
        """检查 LLM 连接"""
        # 保存当前滚动位置
        scroll_position = self.verticalScrollBar().value()

        # 获取当前选中的服务
        current_service = LLMServiceEnum(self.llmServiceCard.comboBox.currentText())

        # 获取服务配置
        service_config = self.llm_service_configs.get(current_service)
        if not service_config:
            return

        api_base = (
            service_config["api_base"].lineEdit.text()
            if service_config["api_base"]
            else ""
        )
        api_key = (
            service_config["api_key"].lineEdit.text()
            if service_config["api_key"]
            else ""
        )
        model = (
            service_config["model"].comboBox.currentText()
            if service_config["model"]
            else ""
        )

        # 禁用检查按钮，显示加载状态
        self.checkLLMConnectionCard.button.setEnabled(False)
        self.checkLLMConnectionCard.button.setText(self.tr("正在检查..."))

        # 立即恢复滚动位置（防止按钮状态改变导致的自动滚动）
        self.verticalScrollBar().setValue(scroll_position)

        # 创建并启动线程
        self.connection_thread = LLMConnectionThread(api_base, api_key, model)
        self.connection_thread.finished.connect(self.onConnectionCheckFinished)
        self.connection_thread.error.connect(self.onConnectionCheckError)
        self.connection_thread.start()

    def onConnectionCheckError(self, message):
        """处理连接检查错误事件"""
        self.checkLLMConnectionCard.button.setEnabled(True)
        self.checkLLMConnectionCard.button.setText(self.tr("检查连接"))
        InfoBar.error(
            self.tr("LLM 连接测试错误"),
            message,
            duration=INFOBAR_DURATION_ERROR,
            parent=self,
        )

    def onConnectionCheckFinished(self, is_success, message, models):
        """处理连接检查完成事件"""
        self.checkLLMConnectionCard.button.setEnabled(True)
        self.checkLLMConnectionCard.button.setText(self.tr("检查连接"))

        # 获取当前服务
        current_service = LLMServiceEnum(self.llmServiceCard.comboBox.currentText())

        if models:
            # 更新当前服务的模型列表
            service_config = self.llm_service_configs.get(current_service)
            if service_config and service_config["model"]:
                temp = service_config["model"].comboBox.currentText()
                service_config["model"].setItems(models)
                service_config["model"].comboBox.setCurrentText(temp)

            InfoBar.success(
                self.tr("获取模型列表成功:"),
                self.tr("一共") + str(len(models)) + self.tr("个模型"),
                duration=INFOBAR_DURATION_SUCCESS,
                parent=self,
            )
        if not is_success:
            InfoBar.error(
                self.tr("LLM 连接测试错误"),
                message,
                duration=INFOBAR_DURATION_ERROR,
                parent=self,
            )
        else:
            InfoBar.success(
                self.tr("LLM 连接测试成功"),
                message,
                duration=INFOBAR_DURATION_SUCCESS,
                parent=self,
            )

    def checkUpdate(self):
        webbrowser.open(RELEASE_URL)

    def __onLLMServiceChanged(self, service):
        """处理LLM服务切换事件"""
        current_service = LLMServiceEnum(service)

        # 隐藏所有卡片
        for config in self.llm_service_configs.values():
            for card in config["cards"]:
                card.setVisible(False)

        # 隐藏OPENAI官方API链接卡片
        self.openaiOfficialApiCard.setVisible(False)

        # 显示选中服务的卡片
        if current_service in self.llm_service_configs:
            for card in self.llm_service_configs[current_service]["cards"]:
                card.setVisible(True)

            # 为OLLAMA和LM_STUDIO设置默认API Key
            service_config = self.llm_service_configs[current_service]
            if current_service == LLMServiceEnum.OLLAMA and service_config["api_key"]:
                # 如果API Key为空，设置默认值"ollama"
                if not service_config["api_key"].lineEdit.text():
                    service_config["api_key"].lineEdit.setText("ollama")
            if (
                current_service == LLMServiceEnum.LM_STUDIO
                and service_config["api_key"]
            ):
                # 如果API Key为空，设置默认值 "lm-studio"
                if not service_config["api_key"].lineEdit.text():
                    service_config["api_key"].lineEdit.setText("lm-studio")

            # 如果是OPENAI服务，显示官方API链接卡片
            if current_service == LLMServiceEnum.OPENAI:
                self.openaiOfficialApiCard.setVisible(True)

        # 更新布局
        self.llmGroup.adjustSize()
        self.expandLayout.update()
        self.__updatePresetActionState()

    def __loadLLMPresets(self) -> list[dict[str, Any]]:
        return parse_llm_provider_presets(cfg.get(cfg.llm_provider_presets))

    def __saveLLMPresets(self, presets: list[dict[str, Any]]) -> None:
        cfg.set(
            cfg.llm_provider_presets,
            json.dumps(presets, ensure_ascii=False, sort_keys=True),
        )

    def __refreshLLMPresetItems(self) -> None:
        presets = self.__loadLLMPresets()
        names = [str(item.get("name", "")).strip() for item in presets]
        names = [name for name in names if name]
        active_name = cfg.get(cfg.llm_active_preset_name).strip()

        self._updating_preset_ui = True
        self.llmPresetCard.comboBox.clear()
        self.llmPresetCard.comboBox.addItem("")
        if names:
            self.llmPresetCard.comboBox.addItems(names)

        if active_name and active_name in names:
            self.llmPresetCard.comboBox.setCurrentText(active_name)
        else:
            self.llmPresetCard.comboBox.setCurrentText("")
            cfg.set(cfg.llm_active_preset_name, "")
        self._updating_preset_ui = False
        self.__updatePresetActionState()

    def __updatePresetActionState(self) -> None:
        has_active = bool(self.llmPresetCard.comboBox.currentText().strip())
        self.overwritePresetCard.button.setEnabled(has_active)
        self.deletePresetCard.button.setEnabled(has_active)

    def __getCurrentServiceSnapshot(self) -> dict[str, Any]:
        current_service = LLMServiceEnum(self.llmServiceCard.comboBox.currentText())
        service_config = self.llm_service_configs[current_service]
        _, structured_item = get_provider_param_items(current_service)
        return {
            "provider": current_service.value,
            "api_key": service_config["api_key"].lineEdit.text().strip(),
            "api_base": service_config["api_base"].lineEdit.text().strip(),
            "model": service_config["model"].comboBox.currentText().strip(),
            "llm_extra_params": service_config["extra_params"].lineEdit.text().strip(),
            "use_structured_outputs": bool(cfg.get(structured_item)),
        }

    def __applyPreset(self, preset: dict[str, Any]) -> None:
        provider_text = str(preset.get("provider", "")).strip()
        if not provider_text:
            return

        try:
            provider = LLMServiceEnum(provider_text)
        except ValueError:
            return

        self.llmServiceCard.comboBox.setCurrentText(provider.value)
        service_config = self.llm_service_configs.get(provider)
        if not service_config:
            return

        service_config["api_key"].lineEdit.setText(str(preset.get("api_key", "")))
        service_config["api_base"].lineEdit.setText(str(preset.get("api_base", "")))
        service_config["model"].comboBox.setCurrentText(str(preset.get("model", "")))
        service_config["extra_params"].lineEdit.setText(
            str(preset.get("llm_extra_params", ""))
        )
        _, structured_item = get_provider_param_items(provider)
        cfg.set(structured_item, bool(preset.get("use_structured_outputs", False)))

    def __savePresetByName(self, name: str, overwrite: bool) -> bool:
        clean_name = name.strip()
        if not clean_name:
            InfoBar.warning(
                self.tr("名称为空"),
                self.tr("请输入有效的预设名称"),
                duration=INFOBAR_DURATION_WARNING,
                parent=self,
            )
            return False

        presets = self.__loadLLMPresets()
        existing_index = next(
            (i for i, item in enumerate(presets) if item.get("name") == clean_name),
            -1,
        )
        snapshot = self.__getCurrentServiceSnapshot()
        snapshot["name"] = clean_name

        if existing_index >= 0 and not overwrite:
            InfoBar.warning(
                self.tr("名称重复"),
                self.tr("预设名称已存在，请使用其他名称"),
                duration=INFOBAR_DURATION_WARNING,
                parent=self,
            )
            return False

        if existing_index >= 0:
            presets[existing_index] = snapshot
        else:
            presets.append(snapshot)

        self.__saveLLMPresets(presets)
        cfg.set(cfg.llm_active_preset_name, clean_name)
        self.__refreshLLMPresetItems()
        return True

    def __saveCurrentAsPreset(self) -> None:
        dialog = PresetNameDialog(self)
        if not dialog.exec():
            return

        if self.__savePresetByName(dialog.nameLineEdit.text(), overwrite=False):
            InfoBar.success(
                self.tr("保存成功"),
                self.tr("已保存新的 LLM 预设"),
                duration=INFOBAR_DURATION_SUCCESS,
                parent=self,
            )

    def __overwriteCurrentPreset(self) -> None:
        current_name = self.llmPresetCard.comboBox.currentText().strip()
        if not current_name:
            InfoBar.warning(
                self.tr("未选择预设"),
                self.tr("请先选择需要覆盖的预设"),
                duration=INFOBAR_DURATION_WARNING,
                parent=self,
            )
            return

        if self.__savePresetByName(current_name, overwrite=True):
            InfoBar.success(
                self.tr("覆盖成功"),
                self.tr("已覆盖当前 LLM 预设"),
                duration=INFOBAR_DURATION_SUCCESS,
                parent=self,
            )

    def __deleteCurrentPreset(self) -> None:
        current_name = self.llmPresetCard.comboBox.currentText().strip()
        if not current_name:
            return

        presets = self.__loadLLMPresets()
        presets = [item for item in presets if item.get("name") != current_name]
        self.__saveLLMPresets(presets)
        cfg.set(cfg.llm_active_preset_name, "")
        self.__refreshLLMPresetItems()
        InfoBar.success(
            self.tr("删除成功"),
            self.tr("已删除当前 LLM 预设"),
            duration=INFOBAR_DURATION_SUCCESS,
            parent=self,
        )

    def __onLLMPresetChanged(self, name: str) -> None:
        if self._updating_preset_ui:
            return

        clean_name = name.strip()
        cfg.set(cfg.llm_active_preset_name, clean_name)
        self.__updatePresetActionState()
        if not clean_name:
            return

        preset = next(
            (item for item in self.__loadLLMPresets() if item.get("name") == clean_name),
            None,
        )
        if preset is None:
            return

        self.__applyPreset(preset)

    def __onTranslatorServiceChanged(self, service):
        openai_cards = [
            self.needReflectTranslateCard,
            self.batchSizeCard,
        ]
        deeplx_cards = [self.deeplxEndpointCard]

        all_cards = openai_cards + deeplx_cards
        for card in all_cards:
            card.setVisible(False)

        # 根据选择的服务显示相应的配置卡片
        if service in [TranslatorServiceEnum.DEEPLX.value]:
            for card in deeplx_cards:
                card.setVisible(True)
        elif service in [TranslatorServiceEnum.OPENAI.value]:
            for card in openai_cards:
                card.setVisible(True)

        # 更新布局
        self.translate_serviceGroup.adjustSize()
        self.expandLayout.update()

    def __onTranscribeModelChanged(self, model_name):
        """处理转录模型切换事件"""
        # Whisper API 配置卡片
        whisper_api_cards = [
            self.whisperApiBaseCard,
            self.whisperApiKeyCard,
            self.whisperApiModelCard,
            self.checkWhisperConnectionCard,
        ]

        # 根据选择的模型显示/隐藏 Whisper API 配置
        is_whisper_api = model_name == TranscribeModelEnum.WHISPER_API.value
        for card in whisper_api_cards:
            card.setVisible(is_whisper_api)

        # 更新布局
        self.transcribeGroup.adjustSize()
        self.expandLayout.update()

    def checkWhisperConnection(self):
        """检查 Whisper API 连接"""
        # 保存当前滚动位置
        scroll_position = self.verticalScrollBar().value()

        # 获取配置
        base_url = self.whisperApiBaseCard.lineEdit.text().strip()
        api_key = self.whisperApiKeyCard.lineEdit.text().strip()
        model = self.whisperApiModelCard.comboBox.currentText().strip()

        # 验证必填字段
        if not base_url:
            InfoBar.warning(
                self.tr("配置不完整"),
                self.tr("请输入 Whisper API Base URL"),
                duration=INFOBAR_DURATION_ERROR,
                parent=self,
            )
            return

        if not api_key:
            InfoBar.warning(
                self.tr("配置不完整"),
                self.tr("请输入 Whisper API Key"),
                duration=INFOBAR_DURATION_ERROR,
                parent=self,
            )
            return

        if not model:
            InfoBar.warning(
                self.tr("配置不完整"),
                self.tr("请输入 Whisper 模型名称"),
                duration=INFOBAR_DURATION_ERROR,
                parent=self,
            )
            return

        # 禁用按钮，显示加载状态
        self.checkWhisperConnectionCard.button.setEnabled(False)
        self.checkWhisperConnectionCard.button.setText(self.tr("正在测试..."))

        # 立即恢复滚动位置（防止按钮状态改变导致的自动滚动）
        self.verticalScrollBar().setValue(scroll_position)

        # 创建并启动测试线程
        self.whisper_connection_thread = WhisperConnectionThread(
            base_url, api_key, model
        )
        self.whisper_connection_thread.finished.connect(
            self.onWhisperConnectionCheckFinished
        )
        self.whisper_connection_thread.error.connect(self.onWhisperConnectionCheckError)
        self.whisper_connection_thread.start()

    def onWhisperConnectionCheckFinished(self, success, result):
        """处理 Whisper 连接检查完成事件"""
        # 恢复按钮状态
        self.checkWhisperConnectionCard.button.setEnabled(True)
        self.checkWhisperConnectionCard.button.setText(self.tr("测试 Whisper 连接"))

        if success:
            InfoBar.success(
                self.tr("连接成功"),
                self.tr("Whisper API 连接成功！\n转录结果:") + result,
                duration=INFOBAR_DURATION_SUCCESS,
                parent=self,
            )
        else:
            InfoBar.error(
                self.tr("连接失败"),
                self.tr(f"Whisper API 连接失败！\n{result}"),
                duration=INFOBAR_DURATION_ERROR,
                parent=self,
            )

    def onWhisperConnectionCheckError(self, message):
        """处理 Whisper 连接检查错误事件"""
        # 恢复按钮状态
        self.checkWhisperConnectionCard.button.setEnabled(True)
        self.checkWhisperConnectionCard.button.setText(self.tr("测试 Whisper 连接"))

        InfoBar.error(
            self.tr("测试错误"),
            message,
            duration=INFOBAR_DURATION_ERROR,
            parent=self,
        )


class PresetNameDialog(MessageBoxBase):
    """LLM 预设名称输入对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = BodyLabel(self.tr("保存 LLM 预设"), self)
        self.nameLineEdit = LineEdit(self)
        self.nameLineEdit.setPlaceholderText(self.tr("输入预设名称"))
        self.nameLineEdit.setClearButtonEnabled(True)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.nameLineEdit)

        self.yesButton.setText(self.tr("确定"))
        self.cancelButton.setText(self.tr("取消"))
        self.widget.setMinimumWidth(350)
        self.yesButton.setDisabled(True)
        self.nameLineEdit.textChanged.connect(self.__validateInput)

    def __validateInput(self, text: str) -> None:
        self.yesButton.setEnabled(bool(text.strip()))


class WhisperConnectionThread(QThread):
    """Whisper API 连接测试线程"""

    finished = pyqtSignal(bool, str)
    error = pyqtSignal(str)

    def __init__(self, base_url, api_key, model):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def run(self):
        """执行连接测试"""
        try:
            success, result = check_whisper_connection(
                self.base_url, self.api_key, self.model
            )
            self.finished.emit(success, result)
        except Exception as e:
            self.error.emit(str(e))


class LLMConnectionThread(QThread):
    finished = pyqtSignal(bool, str, list)
    error = pyqtSignal(str)

    def __init__(self, api_base, api_key, model):
        super().__init__()
        self.api_base = api_base
        self.api_key = api_key
        self.model = model

    def run(self):
        """检查 LLM 连接并获取模型列表"""
        try:
            is_success, message = check_llm_connection(
                self.api_base, self.api_key, self.model
            )
            models = get_available_models(self.api_base, self.api_key)
            self.finished.emit(is_success, message, models)
        except Exception as e:
            self.error.emit(str(e))
