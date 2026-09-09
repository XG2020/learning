"""Nekro Agent 自学习插件。

插件会按频道隔离保存学习状态，学习结果默认只影响当前频道的提示词，
不会修改 Nekro 的全局人设或核心数据库结构。
"""

from . import main
from .plugin import plugin

__all__ = ["main", "plugin"]
