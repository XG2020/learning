from pydantic import Field

from nekro_agent.api import i18n
from nekro_agent.api.plugin import ConfigBase, ExtraField, NekroPlugin


plugin = NekroPlugin(
    name="自学习",
    module_name="learning",
    description="从频道消息中学习表达风格、偏好与黑话，并在后续对话中注入经过审查的学习结果",
    version="1.1.0",
    author="xggm",
    url="",
    i18n_name=i18n.i18n_text(zh_CN="自学习", en_US="Self Learning"),
    i18n_description=i18n.i18n_text(
        zh_CN="从频道消息中学习表达风格、偏好与黑话，并在后续对话中注入经过审查的学习结果",
        en_US="Learns conversation style, preferences, and jargon from channel messages",
    ),
    allow_sleep=False,
    sleep_brief="持续采集频道对话并注入已学习的表达风格、偏好和本地黑话。",
    webui_path="/",
)


def _extra(zh_title: str, en_title: str, zh_description: str, en_description: str) -> dict:
    return ExtraField(
        i18n_title=i18n.i18n_text(zh_CN=zh_title, en_US=en_title),
        i18n_description=i18n.i18n_text(zh_CN=zh_description, en_US=en_description),
    ).model_dump()


@plugin.mount_config()
class LearningConfig(ConfigBase):
    ENABLE_LONG_TERM_MEMORY_WRITE: bool = Field(
        default=False,
        title="将学习结果写入长期记忆",
        description="开启后，已批准的学习结果会写入当前频道绑定工作区的 Nekro 长期记忆",
        json_schema_extra=_extra(
            "将学习结果写入长期记忆",
            "Write Learning Results to Long-term Memory",
            "开启后，已批准的学习结果会写入当前频道绑定工作区的 Nekro 长期记忆",
            "Write approved learning results to the Nekro long-term memory of the bound workspace",
        ),
    )
    ENABLE_WEBUI: bool = Field(
        default=True,
        title="启用学习管理 WebUI",
        description="提供频道学习状态、审查结果和数据管理页面",
        json_schema_extra=_extra(
            "启用学习管理 WebUI",
            "Enable Learning Management WebUI",
            "提供频道学习状态、审查结果和数据管理页面",
            "Enable the learning status, review, and data management page",
        ),
    )
    ENABLE_WEBUI_AUTH: bool = Field(
        default=True,
        title="启用 WebUI 密码认证",
        description="启用后，WebUI 管理 API 需要密码；密码为空时为兼容开发环境而免密",
        json_schema_extra=_extra(
            "启用 WebUI 密码认证",
            "Enable WebUI Password Authentication",
            "启用后，WebUI 管理 API 需要密码；密码为空时为兼容开发环境而免密",
            "Require a password for WebUI management APIs; blank password keeps development mode passwordless",
        ),
    )
    WEBUI_PASSWORD: str = Field(
        default="",
        title="WebUI 访问密码",
        description="设置后访问管理 API 必须提供此密码，建议使用高强度随机密码",
        json_schema_extra={
            **_extra(
                "WebUI 访问密码",
                "WebUI Access Password",
                "设置后访问管理 API 必须提供此密码，建议使用高强度随机密码",
                "Password required by the management API; use a strong random password",
            ),
            "is_secret": True,
            "placeholder": "留空表示免密（仅建议本地开发）",
        },
    )
    ENABLE_MESSAGE_CAPTURE: bool = Field(
        default=True,
        title="启用消息采集",
        description="采集符合条件的用户消息作为学习样本",
        json_schema_extra=_extra(
            "启用消息采集",
            "Enable Message Capture",
            "采集符合条件的用户消息作为学习样本",
            "Collect eligible user messages as learning samples",
        ),
    )
    ENABLE_AUTO_LEARNING: bool = Field(
        default=True,
        title="启用自动学习",
        description="样本达到阈值后自动执行学习",
        json_schema_extra=_extra(
            "启用自动学习",
            "Enable Auto Learning",
            "样本达到阈值后自动执行学习",
            "Run learning automatically when enough samples are collected",
        ),
    )
    ENABLE_REALTIME_LEARNING: bool = Field(
        default=False,
        title="启用实时学习",
        description="每条有效消息都尝试触发学习，可能显著增加模型调用",
        json_schema_extra=_extra(
            "启用实时学习",
            "Enable Realtime Learning",
            "每条有效消息都尝试触发学习，可能显著增加模型调用",
            "Try learning after every eligible message; this may increase model usage",
        ),
    )
    ENABLE_LLM_ANALYSIS: bool = Field(
        default=True,
        title="启用 LLM 分析",
        description="使用当前 Nekro 模型提炼风格、偏好和黑话；失败时自动退回规则分析",
        json_schema_extra=_extra(
            "启用 LLM 分析",
            "Enable LLM Analysis",
            "使用当前 Nekro 模型提炼风格、偏好和黑话；失败时自动退回规则分析",
            "Use the active Nekro model and fall back to rule analysis on failure",
        ),
    )
    AUTO_APPLY_REVIEWS: bool = Field(
        default=True,
        title="自动应用学习结果",
        description="关闭后学习结果进入待审查状态，需手动批准才会注入对话",
        json_schema_extra=_extra(
            "自动应用学习结果",
            "Auto Apply Reviews",
            "关闭后学习结果进入待审查状态，需手动批准才会注入对话",
            "When disabled, learned results require manual approval before injection",
        ),
    )
    INJECT_LEARNED_PROFILE: bool = Field(
        default=True,
        title="注入学习结果",
        description="在每轮对话前注入当前频道已经批准的学习结果",
        json_schema_extra=_extra(
            "注入学习结果",
            "Inject Learned Profile",
            "在每轮对话前注入当前频道已经批准的学习结果",
            "Inject approved channel learning results before each conversation",
        ),
    )
    MIN_MESSAGES_FOR_LEARNING: int = Field(default=20, ge=1, le=500, title="学习触发消息数")
    MAX_MESSAGES_PER_BATCH: int = Field(default=80, ge=1, le=500, title="单批最大学习消息数")
    MAX_BUFFERED_MESSAGES: int = Field(default=200, ge=10, le=2000, title="最大待学习消息数")
    LEARNING_INTERVAL_MINUTES: int = Field(default=60, ge=0, le=10080, title="最短学习间隔（分钟）")
    MESSAGE_MIN_LENGTH: int = Field(default=3, ge=1, le=100, title="最短消息长度")
    MESSAGE_MAX_LENGTH: int = Field(default=500, ge=20, le=5000, title="最长消息长度")
    MAX_REVIEWS: int = Field(default=30, ge=1, le=200, title="保留审查记录数")
    MAX_PROMPT_CHARS: int = Field(default=3000, ge=500, le=12000, title="最大注入字符数")
    MODEL_GROUP: str = Field(default="", title="学习模型组", description="留空时使用 Nekro 当前默认模型组")
    TARGET_CHAT_KEYS: list[str] = Field(default_factory=list, title="学习频道白名单")
    CHAT_KEY_BLACKLIST: list[str] = Field(default_factory=list, title="学习频道黑名单")
    SENDER_BLACKLIST: list[str] = Field(default_factory=list, title="发送者黑名单")


config: LearningConfig = plugin.get_config(LearningConfig)
