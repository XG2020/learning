from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .models import LearningSample, ProfileProposal, StyleMetrics


_LOW_INFORMATION_MESSAGES = {
    "嗯",
    "哦",
    "啊",
    "额",
    "好",
    "好的",
    "行",
    "可以",
    "收到",
    "哈哈",
    "哈哈哈",
    "谢谢",
    "1",
    "6",
    "ok",
    "okay",
}
_COMMON_TERMS = {
    "这个",
    "那个",
    "然后",
    "就是",
    "还是",
    "但是",
    "因为",
    "所以",
    "什么",
    "怎么",
    "可以",
    "一个",
    "没有",
    "不是",
    "现在",
    "感觉",
    "已经",
    "自己",
    "我们",
    "你们",
    "他们",
    "really",
    "this",
    "that",
    "with",
    "from",
    "have",
    "just",
}
_COMMAND_PREFIXES = ("/", "!", "！")
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{1,19}|[\u4e00-\u9fff]{2,6}")
_IDENTITY_RE = re.compile(r"(?:用户|发送者|成员|联系人)?\s*(?:昵称|用户名|用户\s*名|姓名|名字|称呼|id)\s*(?:是|为|叫|:|：)?", re.IGNORECASE)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def should_collect_text(text: str, min_length: int, max_length: int) -> bool:
    normalized = normalize_text(text)
    if not normalized or len(normalized) < min_length or len(normalized) > max_length:
        return False
    if normalized.startswith(_COMMAND_PREFIXES):
        return False
    if normalized.casefold() in _LOW_INFORMATION_MESSAGES:
        return False
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", normalized))


def _ratio(matches: int, total: int) -> float:
    return round(matches / total, 3) if total else 0.0


def _common_endings(texts: list[str]) -> list[str]:
    endings: Counter[str] = Counter()
    for text in texts:
        cleaned = text.rstrip("。！？!?~～. ")
        if len(cleaned) >= 2:
            endings[cleaned[-2:]] += 1
    return [ending for ending, count in endings.most_common(5) if count >= 2]


def _common_terms(texts: list[str]) -> list[str]:
    terms: Counter[str] = Counter()
    for text in texts:
        for match in _TERM_RE.findall(text):
            term = match.casefold()
            if term not in _COMMON_TERMS:
                terms[term] += 1
    return [term for term, count in terms.most_common(12) if count >= 2]


def calculate_style_metrics(samples: list[LearningSample]) -> StyleMetrics:
    texts = [sample.text.strip() for sample in samples if sample.text.strip()]
    total = len(texts)
    if not total:
        return StyleMetrics()

    return StyleMetrics(
        sample_count=total,
        average_length=round(sum(len(text) for text in texts) / total, 1),
        short_message_ratio=_ratio(sum(len(text) <= 15 for text in texts), total),
        question_ratio=_ratio(sum("?" in text or "？" in text for text in texts), total),
        exclamation_ratio=_ratio(sum("!" in text or "！" in text for text in texts), total),
        emoji_ratio=_ratio(sum(bool(_EMOJI_RE.search(text)) for text in texts), total),
        multiline_ratio=_ratio(sum("\n" in text for text in texts), total),
        common_endings=_common_endings(texts),
        common_terms=_common_terms(texts),
    )


def build_rule_proposal(samples: list[LearningSample]) -> ProfileProposal:
    metrics = calculate_style_metrics(samples)
    if not metrics.sample_count:
        return ProfileProposal(metrics=metrics)

    style_parts = [f"平均消息长度约 {metrics.average_length:.0f} 字"]
    guidance: list[str] = []
    if metrics.short_message_ratio >= 0.6:
        style_parts.append("偏好短句和快速往返")
        guidance.append("优先使用简洁短句，避免没有必要的长篇展开")
    else:
        style_parts.append("能够接受较完整的说明")
    if metrics.question_ratio >= 0.3:
        style_parts.append("经常以提问推进对话")
        guidance.append("直接回应问题核心，并在信息不足时提出一个明确的追问")
    if metrics.emoji_ratio >= 0.2:
        style_parts.append("会自然使用表情符号")
        guidance.append("可少量沿用频道的表情习惯，但不要刻意堆叠")
    if metrics.multiline_ratio >= 0.2:
        style_parts.append("常使用分行组织内容")
        guidance.append("复杂回答可分段组织，保持易扫描")
    if not guidance:
        guidance.append("保持自然、直接，并与频道当前交流密度一致")

    jargon = {term: "频道中的高频表达，具体含义需结合当前上下文判断" for term in metrics.common_terms[:6]}
    return ProfileProposal(
        style_summary="；".join(style_parts) + "。",
        reply_guidance=guidance,
        jargon=jargon,
        metrics=metrics,
    )


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip().lstrip("\ufeff")
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    if not candidate.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", candidate)
        candidate = match.group(0) if match else candidate
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("学习模型返回值不是 JSON 对象")
    return data


def parse_llm_proposal(raw: str, metrics: StyleMetrics) -> ProfileProposal:
    data = _extract_json_object(raw)
    jargon_raw = data.get("jargon", {})
    jargon: dict[str, str] = {}
    if isinstance(jargon_raw, dict):
        jargon = {str(key).strip(): str(value).strip() for key, value in jargon_raw.items() if str(key).strip()}
    elif isinstance(jargon_raw, list):
        for item in jargon_raw:
            if isinstance(item, dict) and str(item.get("term", "")).strip():
                jargon[str(item["term"]).strip()] = str(item.get("meaning", "")).strip()

    def string_list(key: str) -> list[str]:
        value = data.get(key, [])
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    return ProfileProposal(
        style_summary=str(data.get("style_summary", "")).strip(),
        reply_guidance=string_list("reply_guidance"),
        persona_observations=string_list("persona_observations"),
        user_preferences=string_list("user_preferences"),
        jargon=jargon,
        metrics=metrics,
    )


def sanitize_llm_proposal(proposal: ProfileProposal, samples: list[LearningSample]) -> ProfileProposal:
    """移除模型误识别出的发送者身份信息，避免昵称/用户 ID 进入学习档案。"""
    identities = {
        normalize_text(value).casefold()
        for sample in samples
        for value in (sample.sender_name, sample.sender_id)
        if normalize_text(value)
    }

    def is_identity_item(value: str) -> bool:
        normalized = normalize_text(value)
        folded = normalized.casefold()
        if not normalized:
            return True
        if folded in identities:
            return True
        return bool(_IDENTITY_RE.search(normalized) and any(identity in folded for identity in identities if identity))

    def clean_items(values: list[str]) -> list[str]:
        return [value for value in values if not is_identity_item(value)]

    clean_jargon: dict[str, str] = {}
    for term, meaning in proposal.jargon.items():
        clean_term = normalize_text(term)
        clean_meaning = normalize_text(meaning)
        if not clean_term or clean_term.casefold() in identities:
            continue
        if _IDENTITY_RE.search(clean_meaning) and any(identity in clean_meaning.casefold() for identity in identities if identity):
            continue
        clean_jargon[clean_term] = clean_meaning

    style_summary = proposal.style_summary
    for identity in sorted(identities, key=len, reverse=True):
        if identity:
            style_summary = re.sub(re.escape(identity), "该用户", style_summary, flags=re.IGNORECASE)
    style_summary = _IDENTITY_RE.sub("", style_summary).strip(" ，,；;。.")

    return ProfileProposal(
        style_summary=style_summary,
        reply_guidance=clean_items(proposal.reply_guidance),
        persona_observations=clean_items(proposal.persona_observations),
        user_preferences=clean_items(proposal.user_preferences),
        jargon=clean_jargon,
        metrics=proposal.metrics,
    )


def merge_rule_and_llm(rule: ProfileProposal, llm: ProfileProposal) -> ProfileProposal:
    return ProfileProposal(
        style_summary=llm.style_summary or rule.style_summary,
        reply_guidance=llm.reply_guidance or rule.reply_guidance,
        persona_observations=llm.persona_observations,
        user_preferences=llm.user_preferences,
        jargon={**rule.jargon, **llm.jargon},
        metrics=rule.metrics,
    )


def build_analysis_prompt(samples: list[LearningSample]) -> str:
    conversation = "\n".join(
        f"- 发送者元数据（仅用于区分消息，绝不是学习对象）：{sample.sender_name or sample.sender_id}\n"
        f"  消息正文（只分析这一部分）：{sample.text[:500]}"
        for sample in samples
    )
    return f"""分析以下频道消息，提取可用于后续对话适配的稳定特征。

要求：
1. 只总结有多条样本支持的表达风格、明确偏好和群内黑话。
2. 不推断年龄、性别、健康、政治、宗教、种族、财务等敏感属性。
3. persona_observations 只能描述可观察的交流倾向，不能给用户贴人格或心理诊断标签。
4. reply_guidance 应是可执行、克制的回复建议，不得要求无条件模仿错误信息、攻击性或危险行为。
5. 发送者元数据中的昵称、用户名、用户 ID、称呼都不是消息内容，严禁把它们输出到 style_summary、persona_observations、user_preferences 或 jargon。
6. 不要记录任何用户身份标识；没有证据的字段返回空数组或空对象。

仅返回以下 JSON，不要附加说明：
{{
  "style_summary": "简短风格总结",
  "reply_guidance": ["建议1"],
  "persona_observations": ["可观察的交流倾向"],
  "user_preferences": ["明确表达过的偏好"],
  "jargon": {{"词语": "结合上下文推断的含义"}}
}}

频道消息：
{conversation[:16000]}
"""
