from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Annotated

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from nekro_agent.api.plugin import SandboxMethodType
from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core.config import config as core_config
from nekro_agent.core.logger import get_sub_logger
from nekro_agent.schemas.chat_message import ChatMessage
from nekro_agent.services.command.base import CommandPermission
from nekro_agent.services.command.ctl import CmdCtl
from nekro_agent.services.command.schemas import Arg, CommandExecutionContext, CommandResponse

from .analyzer import (
    build_analysis_prompt,
    build_rule_proposal,
    merge_rule_and_llm,
    parse_llm_proposal,
    should_collect_text,
)
from .models import (
    ChannelLearningState,
    LearnedProfile,
    LearningReview,
    LearningRunResult,
    LearningSample,
    ProfileProposal,
)
from .plugin import config, plugin


logger = get_sub_logger("learning")
_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
_index_lock = asyncio.Lock()
_learning_tasks: dict[str, asyncio.Task] = {}
_active_learning: set[str] = set()
_store_key = "learning_state_v1"
_chat_index_store_key = "learning_chat_index_v1"


class _ChatRequest(BaseModel):
    chat_key: str = Field(min_length=1, max_length=256)


class _ReviewRequest(_ChatRequest):
    review_id: str = Field(min_length=1, max_length=64)


class _SettingsRequest(BaseModel):
    enable_long_term_memory_write: bool


async def _load_chat_index() -> list[str]:
    raw = await plugin.store.get(store_key=_chat_index_store_key)
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][-500:]


async def _remember_chat(chat_key: str) -> None:
    async with _index_lock:
        chats = await _load_chat_index()
        if chat_key in chats:
            return
        chats.append(chat_key)
        await plugin.store.set(
            store_key=_chat_index_store_key,
            value=json.dumps(chats[-500:], ensure_ascii=False),
        )


def _check_webui_auth(request: Request) -> None:
    """校验管理 API 密码；密码为空时保留本地开发免密兼容行为。"""
    if not config.ENABLE_WEBUI_AUTH or not config.WEBUI_PASSWORD:
        return
    supplied = request.headers.get("X-Learning-Password", "")
    authorization = request.headers.get("Authorization", "")
    if not supplied and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not secrets.compare_digest(supplied, config.WEBUI_PASSWORD):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="WebUI access password is required",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _ensure_webui_enabled() -> None:
    if not config.ENABLE_WEBUI:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Learning WebUI is disabled")


async def _write_learning_to_long_term_memory(
    chat_key: str,
    profile: LearnedProfile,
    review_id: str,
) -> bool:
    """将已批准的学习档案写入当前频道绑定工作区的原生长期记忆。"""
    if not config.ENABLE_LONG_TERM_MEMORY_WRITE:
        return False

    from nekro_agent.models.db_chat_channel import DBChatChannel
    from nekro_agent.models.db_mem_paragraph import (
        CognitiveType,
        DBMemParagraph,
        KnowledgeType,
        OriginKind,
    )
    from nekro_agent.services.memory.feature_flags import is_memory_system_enabled

    if not is_memory_system_enabled():
        logger.info("Nekro 长期记忆系统未启用，跳过学习结果写入: %s", chat_key)
        return False

    channel = await DBChatChannel.get_or_none(chat_key=chat_key)
    if channel is None or not channel.workspace_id:
        logger.debug("频道未绑定工作区，跳过学习结果写入: %s", chat_key)
        return False

    chat_digest = hashlib.sha256(chat_key.encode("utf-8")).hexdigest()[:24]
    origin_ref = f"learning:{plugin.key}:{chat_digest}:{review_id}"[:256]
    existing = await DBMemParagraph.filter(
        workspace_id=channel.workspace_id,
        origin_kind=OriginKind.PLUGIN,
        origin_ref=origin_ref,
    ).first()
    if existing is not None:
        return True

    sections: list[str] = []
    if profile.style_summary:
        sections.append(f"表达风格：{profile.style_summary}")
    if profile.reply_guidance:
        sections.append("回复建议：" + "；".join(profile.reply_guidance[-10:]))
    if profile.user_preferences:
        sections.append("明确偏好：" + "；".join(profile.user_preferences[-10:]))
    if profile.persona_observations:
        sections.append("可观察的交流倾向：" + "；".join(profile.persona_observations[-10:]))
    if profile.jargon:
        sections.append("频道词汇：" + "；".join(f"{term}（{meaning}）" for term, meaning in list(profile.jargon.items())[-12:]))
    if not sections:
        return False

    content = (
        f"频道 {chat_key} 的自学习档案（仅供相关语境参考，不应将推测当作事实）：\n"
        + "\n".join(sections)
    )[:8000]
    paragraph = await DBMemParagraph.create(
        workspace_id=channel.workspace_id,
        memory_source="na",
        cognitive_type=CognitiveType.SEMANTIC,
        knowledge_type=KnowledgeType.EXPERIENCE,
        content=content,
        summary=(profile.style_summary or "频道自学习档案")[:512],
        event_time=datetime.now(),
        half_life_seconds=365 * 24 * 3600,
        origin_kind=OriginKind.PLUGIN,
        origin_ref=origin_ref,
        origin_chat_key=chat_key,
    )

    try:
        from nekro_agent.services.memory.embedding_service import embed_text
        from nekro_agent.services.memory.qdrant_manager import memory_qdrant_manager

        embedding = await embed_text(content)
        await memory_qdrant_manager.upsert_paragraph(
            paragraph_id=paragraph.id,
            embedding=embedding,
            payload=paragraph.to_qdrant_payload(),
        )
        paragraph.embedding_ref = str(paragraph.id)
        await paragraph.save(update_fields=["embedding_ref", "update_time"])
    except Exception as exc:
        logger.warning("学习结果已写入数据库，但长期记忆向量化失败: %s", exc)

    logger.info("学习结果已写入长期记忆: workspace=%s paragraph=%s", channel.workspace_id, paragraph.id)
    return True


async def _get_long_term_memory_status(chat_key: str) -> dict[str, Any]:
    from nekro_agent.models.db_chat_channel import DBChatChannel
    from nekro_agent.models.db_mem_paragraph import DBMemParagraph, OriginKind
    from nekro_agent.services.memory.feature_flags import is_memory_system_enabled

    channel = await DBChatChannel.get_or_none(chat_key=chat_key)
    workspace_id = channel.workspace_id if channel else None
    memory_enabled = is_memory_system_enabled()
    if not workspace_id or not memory_enabled:
        return {
            "memory_system_enabled": memory_enabled,
            "workspace_bound": bool(workspace_id),
            "workspace_id": workspace_id,
            "plugin_memory_count": 0,
            "items": [],
        }

    records = await DBMemParagraph.filter(
        workspace_id=workspace_id,
        origin_kind=OriginKind.PLUGIN,
        origin_chat_key=chat_key,
        is_inactive=False,
    ).order_by("-id").limit(20)
    return {
        "memory_system_enabled": memory_enabled,
        "workspace_bound": True,
        "workspace_id": workspace_id,
        "plugin_memory_count": await DBMemParagraph.filter(
            workspace_id=workspace_id,
            origin_kind=OriginKind.PLUGIN,
            origin_chat_key=chat_key,
            is_inactive=False,
        ).count(),
        "items": [
            {
                "id": item.id,
                "summary": item.summary or "频道自学习档案",
                "content": item.content,
                "created_at": item.create_time.isoformat() if item.create_time else None,
            }
            for item in records
        ],
    }


async def _load_state(chat_key: str) -> ChannelLearningState:
    raw = await plugin.store.get(chat_key=chat_key, store_key=_store_key)
    if not raw:
        return ChannelLearningState()
    try:
        return ChannelLearningState.model_validate_json(raw)
    except Exception:
        logger.warning("频道学习数据损坏，已从空状态恢复: %s", chat_key, exc_info=True)
        return ChannelLearningState()


async def _save_state(chat_key: str, state: ChannelLearningState) -> None:
    state.samples = state.samples[-config.MAX_BUFFERED_MESSAGES :]
    state.recent_message_ids = state.recent_message_ids[-200:]
    state.reviews = state.reviews[-config.MAX_REVIEWS :]
    await plugin.store.set(
        chat_key=chat_key,
        store_key=_store_key,
        value=state.model_dump_json(),
    )


def _allowed_chat(chat_key: str) -> bool:
    whitelist = {item.strip() for item in config.TARGET_CHAT_KEYS if item.strip()}
    blacklist = {item.strip() for item in config.CHAT_KEY_BLACKLIST if item.strip()}
    return (not whitelist or chat_key in whitelist) and chat_key not in blacklist


def _allowed_sender(sender_id: str) -> bool:
    return sender_id.strip() not in {item.strip() for item in config.SENDER_BLACKLIST if item.strip()}


async def _call_learning_llm(prompt: str) -> str:
    # 延迟导入，避免模型客户端的可选依赖阻止插件本身加载。
    from nekro_agent.services.agent.openai import gen_openai_chat_response

    model_group_name = config.MODEL_GROUP.strip() or core_config.USE_MODEL_GROUP
    model_group = core_config.get_model_group_info(model_group_name)
    response = await gen_openai_chat_response(
        model=model_group.CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一个谨慎的对话风格分析助手。"},
            {"role": "user", "content": prompt},
        ],
        api_key=model_group.API_KEY,
        base_url=model_group.BASE_URL,
        temperature=0.2,
        max_tokens=1800,
    )
    return response.response_content


async def _analyze_samples(samples: list[LearningSample]) -> tuple[ProfileProposal, str, str]:
    rule_proposal = build_rule_proposal(samples)
    if not config.ENABLE_LLM_ANALYSIS:
        return rule_proposal, "rules", ""
    try:
        llm_proposal = parse_llm_proposal(
            await _call_learning_llm(build_analysis_prompt(samples)),
            rule_proposal.metrics,
        )
        return merge_rule_and_llm(rule_proposal, llm_proposal), "mixed", ""
    except Exception as exc:
        logger.warning("学习 LLM 分析失败，退回规则分析: %s", exc)
        return rule_proposal, "rules", str(exc)


def _merge_unique(existing: list[str], additions: list[str], limit: int = 30) -> list[str]:
    result = list(existing)
    seen = {item.casefold() for item in result}
    for item in additions:
        normalized = item.strip()
        if normalized and normalized.casefold() not in seen:
            result.append(normalized)
            seen.add(normalized.casefold())
    return result[-limit:]


def _apply_proposal(profile: LearnedProfile, proposal: ProfileProposal, sample_count: int) -> None:
    if proposal.style_summary:
        profile.style_summary = proposal.style_summary
    profile.reply_guidance = _merge_unique(profile.reply_guidance, proposal.reply_guidance)
    profile.persona_observations = _merge_unique(profile.persona_observations, proposal.persona_observations)
    profile.user_preferences = _merge_unique(profile.user_preferences, proposal.user_preferences)
    profile.jargon.update({key: value for key, value in proposal.jargon.items() if key and value})
    profile.metrics = proposal.metrics
    profile.source_sample_count += sample_count
    profile.revision += 1
    profile.updated_at = int(time.time())


async def _learn_chat(chat_key: str, force: bool = False) -> LearningRunResult:
    async with _locks[chat_key]:
        if chat_key in _active_learning:
            return LearningRunResult(success=False, message="当前频道已有学习任务正在运行。")
        state = await _load_state(chat_key)
        if not state.samples:
            return LearningRunResult(success=False, message="当前没有待学习的消息。")
        if not force and len(state.samples) < config.MIN_MESSAGES_FOR_LEARNING:
            return LearningRunResult(
                success=False,
                message=f"待学习消息不足，还需要 {config.MIN_MESSAGES_FOR_LEARNING - len(state.samples)} 条。",
                sample_count=len(state.samples),
            )
        if (
            not force
            and config.LEARNING_INTERVAL_MINUTES
            and state.stats.last_learning_at
            and time.time() - state.stats.last_learning_at < config.LEARNING_INTERVAL_MINUTES * 60
        ):
            return LearningRunResult(success=False, message="距离上次学习的间隔尚未达到配置阈值。")
        batch = state.samples[: config.MAX_MESSAGES_PER_BATCH]
        _active_learning.add(chat_key)

    try:
        # 模型请求不持有频道锁，避免阻塞新消息采集。
        proposal, source, llm_error = await _analyze_samples(batch)
        async with _locks[chat_key]:
            latest = await _load_state(chat_key)
            batch_ids = {sample.message_id for sample in batch}
            latest.samples = [sample for sample in latest.samples if sample.message_id not in batch_ids]
            review = LearningReview(
                status="approved" if config.AUTO_APPLY_REVIEWS else "pending",
                sample_count=len(batch),
                proposal=proposal,
                analysis_source=source,
            )
            if config.AUTO_APPLY_REVIEWS:
                _apply_proposal(latest.active_profile, proposal, len(batch))
                review.decided_at = int(time.time())
            latest.reviews.append(review)
            latest.stats.total_learning_runs += 1
            latest.stats.last_learning_at = int(time.time())
            if llm_error:
                latest.stats.total_llm_failures += 1
                latest.stats.last_error = llm_error[:500]
            await _save_state(chat_key, latest)

        memory_written = False
        if config.AUTO_APPLY_REVIEWS:
            try:
                memory_written = await _write_learning_to_long_term_memory(
                    chat_key,
                    latest.active_profile,
                    review.review_id,
                )
            except Exception:
                logger.exception("自动写入长期记忆失败: %s", chat_key)

        action = "已应用" if config.AUTO_APPLY_REVIEWS else "已提交审查"
        if memory_written:
            action += "，并写入长期记忆"
        return LearningRunResult(
            success=True,
            message=f"本次学习处理 {len(batch)} 条消息，学习结果{action}。",
            review_id=review.review_id,
            sample_count=len(batch),
            analysis_source=source,
        )
    finally:
        _active_learning.discard(chat_key)


def _schedule_learning(chat_key: str) -> None:
    current = _learning_tasks.get(chat_key)
    if current and not current.done():
        return

    task = asyncio.create_task(_learn_chat(chat_key))
    _learning_tasks[chat_key] = task

    def _done(done_task: asyncio.Task) -> None:
        _learning_tasks.pop(chat_key, None)
        try:
            result = done_task.result()
            if result.success:
                logger.info("频道学习完成: chat_key=%s, samples=%s", chat_key, result.sample_count)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("频道后台学习任务失败: %s", chat_key)

    task.add_done_callback(_done)


@plugin.mount_on_user_message()
async def on_user_message(_ctx: AgentCtx, message: ChatMessage):
    """采集用户消息并按配置启动后台学习。"""
    chat_key = message.chat_key or _ctx.chat_key
    if not config.ENABLE_MESSAGE_CAPTURE or not _allowed_chat(chat_key):
        return None
    if str(message.sender_id) == "-1" or message.is_recalled or not _allowed_sender(message.sender_id):
        return None
    if not should_collect_text(message.content_text, config.MESSAGE_MIN_LENGTH, config.MESSAGE_MAX_LENGTH):
        return None

    async with _locks[chat_key]:
        state = await _load_state(chat_key)
        if not state.capture_enabled:
            return None
        if message.message_id and message.message_id in state.recent_message_ids:
            return None
        if message.message_id:
            state.recent_message_ids.append(message.message_id)
        state.stats.total_messages_collected += 1
        state.samples.append(
            LearningSample(
                message_id=message.message_id or f"local-{time.time_ns()}",
                sender_id=str(message.sender_id),
                sender_name=message.sender_nickname or message.sender_name or str(message.sender_id),
                text=message.content_text.strip(),
                timestamp=message.send_timestamp or int(time.time()),
            )
        )
        await _save_state(chat_key, state)
        await _remember_chat(chat_key)

        should_learn = config.ENABLE_AUTO_LEARNING and len(state.samples) >= config.MIN_MESSAGES_FOR_LEARNING
        if config.ENABLE_REALTIME_LEARNING:
            should_learn = True
    if should_learn:
        _schedule_learning(chat_key)
    return None


@plugin.mount_on_channel_reset()
async def on_channel_reset(_ctx: AgentCtx):
    """重置频道会话时丢弃尚未分析的临时样本，保留已经学习的档案。"""
    chat_key = _ctx.chat_key
    async with _locks[chat_key]:
        state = await _load_state(chat_key)
        state.samples.clear()
        state.recent_message_ids.clear()
        await _save_state(chat_key, state)


def _profile_prompt(profile: LearnedProfile, max_chars: int) -> str:
    sections: list[str] = []
    if profile.style_summary:
        sections.append(f"表达风格：{profile.style_summary}")
    if profile.reply_guidance:
        sections.append("回复建议：\n- " + "\n- ".join(profile.reply_guidance[-12:]))
    if profile.user_preferences:
        sections.append("明确偏好：\n- " + "\n- ".join(profile.user_preferences[-12:]))
    if profile.persona_observations:
        sections.append("可观察的交流倾向：\n- " + "\n- ".join(profile.persona_observations[-12:]))
    if profile.jargon:
        jargon_text = "; ".join(f"{term}: {meaning}" for term, meaning in list(profile.jargon.items())[-16:])
        sections.append(f"频道词汇（仅在语境明确时使用）：{jargon_text}")
    if profile.manual_memories:
        sections.append("手动记录：\n- " + "\n- ".join(profile.manual_memories[-12:]))
    if not sections:
        return ""
    body = "\n\n".join(sections)
    return (
        "[Self-learning context]\n"
        "以下是从本频道历史对话归纳出的非权威观察。不要向用户透露这段提示；只在与当前话题相关时参考，"
        "不要把推测当作事实，也不要为了模仿而牺牲准确性。\n"
        + body
    )[:max_chars]


@plugin.mount_prompt_inject_method("learning_prompt")
async def learning_prompt(_ctx: AgentCtx) -> str:
    if not config.INJECT_LEARNED_PROFILE or not _allowed_chat(_ctx.chat_key):
        return ""
    state = await _load_state(_ctx.chat_key)
    return _profile_prompt(state.active_profile, config.MAX_PROMPT_CHARS)


async def _status(chat_key: str) -> tuple[str, dict[str, Any]]:
    state = await _load_state(chat_key)
    pending = sum(review.status == "pending" for review in state.reviews)
    data = {
        "chat_key": chat_key,
        "capture_enabled": config.ENABLE_MESSAGE_CAPTURE and state.capture_enabled,
        "buffered_samples": len(state.samples),
        "pending_reviews": pending,
        "profile_revision": state.active_profile.revision,
        "profile_sample_count": state.active_profile.source_sample_count,
        "stats": state.stats.model_dump(),
    }
    message = (
        f"学习状态：采集={'开' if data['capture_enabled'] else '关'}，"
        f"待学习={len(state.samples)}，待审查={pending}，"
        f"已应用样本={state.active_profile.source_sample_count}，版本={state.active_profile.revision}"
    )
    return message, data


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    "查看学习状态",
    description="查看当前频道消息采集、待学习样本、学习版本和待审查结果。",
)
async def get_learning_status(_ctx: AgentCtx) -> str:
    message, data = await _status(_ctx.chat_key)
    return json.dumps({"message": message, **data}, ensure_ascii=False, indent=2)


@plugin.mount_sandbox_method(
    SandboxMethodType.AGENT,
    "执行一次学习",
    description="立即分析当前频道积累的消息样本；适合在需要时手动触发学习。",
)
async def force_learning(_ctx: AgentCtx) -> str:
    result = await _learn_chat(_ctx.chat_key, force=True)
    return result.message


@plugin.mount_sandbox_method(
    SandboxMethodType.BEHAVIOR,
    "审批学习结果",
    description="批准当前频道的一条待审查学习结果；review_id 留空时批准最新一条。",
)
async def approve_learning(_ctx: AgentCtx, review_id: str = "") -> str:
    async with _locks[_ctx.chat_key]:
        state = await _load_state(_ctx.chat_key)
        review = next(
            (
                item
                for item in reversed(state.reviews)
                if item.status == "pending" and (not review_id or item.review_id == review_id)
            ),
            None,
        )
        if review is None:
            return "没有找到匹配的待审查学习结果。"
        _apply_proposal(state.active_profile, review.proposal, review.sample_count)
        review.status = "approved"
        review.decided_at = int(time.time())
        await _save_state(_ctx.chat_key, state)
        approved_review_id = review.review_id
        active_profile = state.active_profile
    memory_written = False
    try:
        memory_written = await _write_learning_to_long_term_memory(
            _ctx.chat_key,
            active_profile,
            approved_review_id,
        )
    except Exception:
        logger.exception("手动批准后写入长期记忆失败: %s", _ctx.chat_key)
    suffix = "并写入长期记忆" if memory_written else ""
    return f"学习结果 {approved_review_id} 已批准并应用{suffix}。"


@plugin.mount_sandbox_method(
    SandboxMethodType.BEHAVIOR,
    "拒绝学习结果",
    description="拒绝当前频道的一条待审查学习结果；review_id 留空时拒绝最新一条。",
)
async def reject_learning(_ctx: AgentCtx, review_id: str = "") -> str:
    async with _locks[_ctx.chat_key]:
        state = await _load_state(_ctx.chat_key)
        review = next(
            (
                item
                for item in reversed(state.reviews)
                if item.status == "pending" and (not review_id or item.review_id == review_id)
            ),
            None,
        )
        if review is None:
            return "没有找到匹配的待审查学习结果。"
        review.status = "rejected"
        review.decided_at = int(time.time())
        await _save_state(_ctx.chat_key, state)
        return f"学习结果 {review.review_id} 已拒绝。"


@plugin.mount_sandbox_method(
    SandboxMethodType.BEHAVIOR,
    "记住学习信息",
    description="手动向当前频道的学习档案添加一条明确的偏好、规则或记忆。",
)
async def remember_learning(_ctx: AgentCtx, content: str, category: str = "memory") -> str:
    text = content.strip()
    if not text or len(text) > 1000:
        raise ValueError("记忆内容不能为空且不能超过 1000 字符。")
    category_text = category.strip() or "memory"
    async with _locks[_ctx.chat_key]:
        state = await _load_state(_ctx.chat_key)
        state.active_profile.manual_memories = _merge_unique(
            state.active_profile.manual_memories,
            [f"[{category_text}] {text}"],
            limit=50,
        )
        state.active_profile.revision += 1
        state.active_profile.updated_at = int(time.time())
        await _save_state(_ctx.chat_key, state)
        active_profile = state.active_profile
    memory_written = False
    if config.ENABLE_LONG_TERM_MEMORY_WRITE:
        try:
            memory_written = await _write_learning_to_long_term_memory(
                _ctx.chat_key,
                active_profile,
                f"manual-{active_profile.revision}",
            )
        except Exception:
            logger.exception("手动记忆写入长期记忆失败: %s", _ctx.chat_key)
    return "已记录到当前频道的学习档案。" + ("并同步到长期记忆。" if memory_written else "")


@plugin.mount_sandbox_method(
    SandboxMethodType.BEHAVIOR,
    "清理学习数据",
    description="清空当前频道的待学习样本、学习档案和审查记录；必须显式传入 confirm=true。",
)
async def clear_learning(_ctx: AgentCtx, confirm: bool = False) -> str:
    if not confirm:
        return "为避免误删，请再次调用并传入 confirm=true。"
    async with _locks[_ctx.chat_key]:
        await plugin.store.delete(chat_key=_ctx.chat_key, store_key=_store_key)
    return "当前频道的学习数据已清理。"


async def _command_ctx(context: CommandExecutionContext) -> AgentCtx:
    return await AgentCtx.create_by_chat_key(context.chat_key)


async def _set_capture(chat_key: str, enabled: bool) -> None:
    async with _locks[chat_key]:
        state = await _load_state(chat_key)
        state.capture_enabled = enabled
        await _save_state(chat_key, state)


@plugin.mount_command(
    name="learning_status",
    description="查看当前频道的自学习状态",
    aliases=["学习状态"],
    permission=CommandPermission.SUPER_USER,
    usage="learning_status",
    category="自学习",
)
async def learning_status_command(context: CommandExecutionContext) -> CommandResponse:
    message, data = await _status(context.chat_key)
    return CmdCtl.success(message, data=data)


@plugin.mount_command(
    name="learning_force",
    description="立即执行一次当前频道的自学习",
    aliases=["强制学习", "force_learning"],
    permission=CommandPermission.SUPER_USER,
    usage="learning_force",
    category="自学习",
)
async def learning_force_command(context: CommandExecutionContext) -> CommandResponse:
    result = await _learn_chat(context.chat_key, force=True)
    return CmdCtl.success(result.message, data=result.model_dump()) if result.success else CmdCtl.failed(result.message)


@plugin.mount_command(
    name="learning_capture",
    description="开启或关闭当前频道的消息采集",
    aliases=["学习采集"],
    permission=CommandPermission.SUPER_USER,
    usage="learning_capture <on|off>",
    category="自学习",
)
async def learning_capture_command(context: CommandExecutionContext, enabled: str = "on") -> CommandResponse:
    value = enabled.strip().lower() in {"on", "true", "1", "开启", "开"}
    await _set_capture(context.chat_key, value)
    return CmdCtl.success(f"当前频道消息采集已{'开启' if value else '关闭'}。", data={"enabled": value, "chat_key": context.chat_key})


@plugin.mount_command(
    name="start_learning",
    description="开启当前频道的消息采集和自动学习",
    aliases=["开始学习"],
    permission=CommandPermission.SUPER_USER,
    usage="start_learning",
    category="自学习",
)
async def start_learning_command(context: CommandExecutionContext) -> CommandResponse:
    await _set_capture(context.chat_key, True)
    return CmdCtl.success("当前频道自学习已开启。")


@plugin.mount_command(
    name="stop_learning",
    description="停止当前频道的消息采集",
    aliases=["停止学习"],
    permission=CommandPermission.SUPER_USER,
    usage="stop_learning",
    category="自学习",
)
async def stop_learning_command(context: CommandExecutionContext) -> CommandResponse:
    await _set_capture(context.chat_key, False)
    return CmdCtl.success("当前频道自学习已停止。")


@plugin.mount_command(
    name="learning_approve",
    description="批准一条待审查学习结果",
    aliases=["批准学习"],
    permission=CommandPermission.SUPER_USER,
    usage="learning_approve [review_id]",
    category="自学习",
)
async def learning_approve_command(context: CommandExecutionContext, review_id: str = "") -> CommandResponse:
    result = await approve_learning(await _command_ctx(context), review_id)
    return CmdCtl.success(result) if "已批准" in result else CmdCtl.failed(result)


@plugin.mount_command(
    name="learning_reject",
    description="拒绝一条待审查学习结果",
    aliases=["拒绝学习"],
    permission=CommandPermission.SUPER_USER,
    usage="learning_reject [review_id]",
    category="自学习",
)
async def learning_reject_command(context: CommandExecutionContext, review_id: str = "") -> CommandResponse:
    result = await reject_learning(await _command_ctx(context), review_id)
    return CmdCtl.success(result) if "已拒绝" in result else CmdCtl.failed(result)


@plugin.mount_command(
    name="remember",
    description="向当前频道学习档案手动添加一条记忆",
    aliases=["记住"],
    permission=CommandPermission.SUPER_USER,
    usage="remember <内容>",
    category="自学习",
)
async def remember_command(
    context: CommandExecutionContext,
    content: Annotated[str, Arg("要记录的内容", positional=True, greedy=True)] = "",
) -> CommandResponse:
    if not content.strip():
        return CmdCtl.failed("请输入要记录的内容。")
    result = await remember_learning(await _command_ctx(context), content)
    return CmdCtl.success(result)


@plugin.mount_cleanup_method()
async def clean_up():
    tasks = list(_learning_tasks.values())
    _learning_tasks.clear()
    _active_learning.clear()
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _locks.clear()


@plugin.mount_router()
def create_router() -> APIRouter:
    """创建学习管理 API；具体页面由插件的 webui_path 提供。"""
    router = APIRouter()

    @router.get("/", include_in_schema=False)
    async def webui_index() -> FileResponse:
        _ensure_webui_enabled()
        return FileResponse(Path(__file__).parent / "webui" / "index.html", media_type="text/html")

    @router.get("/api/health")
    async def webui_health(request: Request) -> dict[str, Any]:
        _ensure_webui_enabled()
        _check_webui_auth(request)
        return {"status": "ok", "plugin": plugin.key, "auth_required": bool(config.ENABLE_WEBUI_AUTH and config.WEBUI_PASSWORD)}

    @router.get("/api/settings")
    async def webui_settings(request: Request, chat_key: str = "") -> dict[str, Any]:
        _ensure_webui_enabled()
        _check_webui_auth(request)
        memory_status = await _get_long_term_memory_status(chat_key.strip()) if chat_key.strip() else {
            "memory_system_enabled": False,
            "workspace_bound": False,
            "workspace_id": None,
            "plugin_memory_count": 0,
            "items": [],
        }
        return {
            "enable_long_term_memory_write": config.ENABLE_LONG_TERM_MEMORY_WRITE,
            "memory_system_available": memory_status["memory_system_enabled"] and memory_status["workspace_bound"],
            "workspace_id": memory_status["workspace_id"],
        }

    @router.post("/api/settings")
    async def webui_update_settings(request: Request, payload: _SettingsRequest) -> dict[str, Any]:
        _ensure_webui_enabled()
        _check_webui_auth(request)
        config.ENABLE_LONG_TERM_MEMORY_WRITE = payload.enable_long_term_memory_write
        plugin.save_config(config)
        return {"message": "长期记忆写入开关已更新。", "enable_long_term_memory_write": config.ENABLE_LONG_TERM_MEMORY_WRITE}

    @router.get("/api/chats")
    async def webui_chats(request: Request) -> dict[str, Any]:
        _ensure_webui_enabled()
        _check_webui_auth(request)
        chat_keys = await _load_chat_index()
        chats: list[dict[str, Any]] = []
        for chat_key in chat_keys:
            state = await _load_state(chat_key)
            chats.append(
                {
                    "chat_key": chat_key,
                    "buffered_samples": len(state.samples),
                    "profile_revision": state.active_profile.revision,
                    "pending_reviews": sum(review.status == "pending" for review in state.reviews),
                }
            )
        return {"chats": chats}

    @router.get("/api/state")
    async def webui_state(request: Request, chat_key: str) -> dict[str, Any]:
        _ensure_webui_enabled()
        _check_webui_auth(request)
        if not _allowed_chat(chat_key):
            raise HTTPException(status_code=403, detail="This chat is outside the configured learning scope")
        message, state_data = await _status(chat_key)
        state = await _load_state(chat_key)
        return {
            "message": message,
            "state": state_data,
            "profile": state.active_profile.model_dump(),
            "reviews": [review.model_dump() for review in state.reviews],
            "long_term_memory": await _get_long_term_memory_status(chat_key),
        }

    @router.post("/api/learn")
    async def webui_learn(request: Request, payload: _ChatRequest) -> dict[str, Any]:
        _ensure_webui_enabled()
        _check_webui_auth(request)
        result = await _learn_chat(payload.chat_key, force=True)
        return result.model_dump()

    @router.post("/api/reviews/approve")
    async def webui_approve(request: Request, payload: _ReviewRequest) -> dict[str, Any]:
        _ensure_webui_enabled()
        _check_webui_auth(request)
        result = await approve_learning(await AgentCtx.create_by_chat_key(payload.chat_key), payload.review_id)
        if "已批准" not in result:
            raise HTTPException(status_code=404, detail=result)
        return {"message": result}

    @router.post("/api/reviews/reject")
    async def webui_reject(request: Request, payload: _ReviewRequest) -> dict[str, Any]:
        _ensure_webui_enabled()
        _check_webui_auth(request)
        result = await reject_learning(await AgentCtx.create_by_chat_key(payload.chat_key), payload.review_id)
        if "已拒绝" not in result:
            raise HTTPException(status_code=404, detail=result)
        return {"message": result}

    @router.post("/api/memory/sync")
    async def webui_sync_memory(request: Request, payload: _ChatRequest) -> dict[str, Any]:
        _ensure_webui_enabled()
        _check_webui_auth(request)
        state = await _load_state(payload.chat_key)
        if not state.active_profile.revision:
            raise HTTPException(status_code=400, detail="当前频道还没有可写入的学习档案")
        written = await _write_learning_to_long_term_memory(
            payload.chat_key,
            state.active_profile,
            f"sync-{state.active_profile.revision}",
        )
        if not written:
            raise HTTPException(status_code=400, detail="当前频道未绑定工作区，或长期记忆系统未启用")
        return {"message": "当前学习档案已同步到长期记忆。"}

    @router.post("/api/clear")
    async def webui_clear(request: Request, payload: _ChatRequest) -> dict[str, Any]:
        _ensure_webui_enabled()
        _check_webui_auth(request)
        async with _locks[payload.chat_key]:
            await plugin.store.delete(chat_key=payload.chat_key, store_key=_store_key)
        return {"message": "当前频道的学习数据已清理。"}

    return router
