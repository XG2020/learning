# 自学习插件

这是一个独立的 Nekro Agent 插件，参考 `self_learning` 的核心闭环实现对话风格学习功能，使用 Nekro 原生插件接口。

- `mount_on_user_message` 采集频道用户消息，并过滤命令、机器人消息、短噪声和黑名单消息；
- 按频道持久化待学习样本、学习档案和审查记录，使用 `PluginStore`，无需迁移数据库；
- 规则分析提供稳定降级能力，可选使用 Nekro 当前模型提炼表达风格、明确偏好和频道黑话；
- 自动学习或手动触发学习；学习结果可自动应用，也可关闭自动应用后手动批准；
- `mount_prompt_inject_method` 在后续对话前注入已应用的学习结果；
- 提供 Agent 工具和超级用户命令：状态、强制学习、采集开关、批准/拒绝、手动记忆和清理。

## 安装
1. 在NA后台插件市场下载，自动安装。
2. 将整个 `learning` 文件夹复制到运行实例的本地插件目录：

```text
data/nekro_agent/plugins/workdir/learning/
```

## 配置

配置文件会自动生成到：

```text
data/nekro_agent/plugin_data/xggm.learning/config.yaml
```

默认每积累 20 条有效消息执行一次学习，默认自动应用结果。关闭 `AUTO_APPLY_REVIEWS` 后，结果会进入待审查列表，需要使用 `learning_approve` 或 Agent 工具批准。

`MODEL_GROUP` 留空时使用 Nekro 当前默认模型组。LLM 调用失败不会阻塞消息采集，会回退到规则分析。

常用超级用户命令：`learning_status`、`start_learning`、`stop_learning`、`force_learning`、`learning_approve`、`learning_reject`、`remember`。

## WebUI

启用插件后，管理页地址为：

```text
/plugins/xggm.learning/
```

`ENABLE_WEBUI` 控制管理页和 API 是否启用；`ENABLE_WEBUI_AUTH` 控制 API 密码认证；`WEBUI_PASSWORD` 设置访问密码。密码为空时为免密兼容模式，正式部署时应设置强密码。页面使用当前浏览器会话保存密码，不会把密码写入 URL。

管理页支持：

- 切换频道并查看待学习样本、档案版本和已应用样本数；
- 手动触发一次学习；
- 查看待审查结果并批准或拒绝；
- 清理指定频道的学习数据；
- 开关“写入长期记忆”，手动将当前档案同步到绑定工作区。

开启 `ENABLE_LONG_TERM_MEMORY_WRITE` 后，自动批准或手动批准的档案会写入 Nekro 原生长期记忆。只有绑定了工作区且 Nekro 长期记忆总开关开启时才会写入；未满足条件时仍保留频道学习档案，不会丢失结果。

## 与 Nekro 原生记忆的关系

Nekro 自带的工作区记忆负责长期事实和事件沉淀；本插件负责频道级表达风格、明确偏好和本地词汇。两者互不修改对方的数据，也不会自动改写全局人设。
