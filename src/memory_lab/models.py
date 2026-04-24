from __future__ import annotations

from typing import Literal, TypedDict


MemorySource = Literal["short_term", "long_term", "episodic", "semantic"]


class RetrievedItem(TypedDict):
    source: MemorySource
    content: str
    score: float
    priority: int
    metadata: dict


class AgentState(TypedDict, total=False):
    session_id: str
    user_input: str
    route_targets: list[MemorySource]
    route_reason: str
    retrieved_items: list[RetrievedItem]
    prompt_context: str
    answer: str
    token_breakdown: dict[str, int]
    used_sources: list[MemorySource]
    memory_hits: dict[str, bool]
