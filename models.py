from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class LearningSample(BaseModel):
    message_id: str
    sender_id: str
    sender_name: str
    text: str
    timestamp: int


class StyleMetrics(BaseModel):
    sample_count: int = 0
    average_length: float = 0.0
    short_message_ratio: float = 0.0
    question_ratio: float = 0.0
    exclamation_ratio: float = 0.0
    emoji_ratio: float = 0.0
    multiline_ratio: float = 0.0
    common_endings: list[str] = Field(default_factory=list)
    common_terms: list[str] = Field(default_factory=list)


class ProfileProposal(BaseModel):
    style_summary: str = ""
    reply_guidance: list[str] = Field(default_factory=list)
    persona_observations: list[str] = Field(default_factory=list)
    user_preferences: list[str] = Field(default_factory=list)
    jargon: dict[str, str] = Field(default_factory=dict)
    metrics: StyleMetrics = Field(default_factory=StyleMetrics)


class LearnedProfile(ProfileProposal):
    revision: int = 0
    source_sample_count: int = 0
    updated_at: int = 0
    manual_memories: list[str] = Field(default_factory=list)


class LearningReview(BaseModel):
    review_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: Literal["pending", "approved", "rejected"] = "pending"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    decided_at: int = 0
    sample_count: int = 0
    proposal: ProfileProposal = Field(default_factory=ProfileProposal)
    analysis_source: Literal["llm", "rules", "mixed"] = "rules"


class LearningStats(BaseModel):
    total_messages_collected: int = 0
    total_messages_filtered: int = 0
    total_learning_runs: int = 0
    total_llm_failures: int = 0
    last_learning_at: int = 0
    last_error: str = ""


class ChannelLearningState(BaseModel):
    capture_enabled: bool = True
    samples: list[LearningSample] = Field(default_factory=list)
    recent_message_ids: list[str] = Field(default_factory=list)
    active_profile: LearnedProfile = Field(default_factory=LearnedProfile)
    reviews: list[LearningReview] = Field(default_factory=list)
    stats: LearningStats = Field(default_factory=LearningStats)


class LearningRunResult(BaseModel):
    success: bool
    message: str
    review_id: str = ""
    sample_count: int = 0
    analysis_source: str = ""
