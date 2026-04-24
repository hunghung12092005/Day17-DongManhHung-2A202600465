from __future__ import annotations

from memory_lab.models import MemorySource


class MemoryRouter:
    LONG_TERM_HINTS = (
        "sở thích",
        "thích",
        "ưu tiên",
        "favorite",
        "prefer",
        "profile",
        "hồ sơ",
        "tôi là ai",
        "nhớ tôi",
        "còn nhớ",
        "mục tiêu dự án",
    )
    EPISODIC_HINTS = (
        "lần trước",
        "trước đây",
        "kinh nghiệm",
        "đã từng",
        "history",
        "sự cố",
        "incident",
        "bài học",
    )
    SEMANTIC_HINTS = (
        "giải thích",
        "là gì",
        "how",
        "what",
        "tại sao",
        "kiến thức",
        "so sánh",
        "langgraph",
        "redis",
        "chroma",
        "token",
        "eviction",
        "context window",
        "ưu tiên giữ",
    )

    def route(self, user_input: str) -> tuple[list[MemorySource], str]:
        text = user_input.lower()
        targets: list[MemorySource] = ["short_term"]
        reasons: list[str] = ["always keep short-term conversation context"]

        if any(hint in text for hint in self.LONG_TERM_HINTS):
            targets.append("long_term")
            reasons.append("question refers to user preferences/profile")
        if any(hint in text for hint in self.EPISODIC_HINTS):
            targets.append("episodic")
            reasons.append("question refers to past experience/incidents")
        if any(hint in text for hint in self.SEMANTIC_HINTS):
            targets.append("semantic")
            reasons.append("question is factual or knowledge-oriented")

        if targets == ["short_term"]:
            targets.append("semantic")
            reasons.append("default fallback is semantic knowledge retrieval")

        # Preserve order while removing duplicates.
        deduped = list(dict.fromkeys(targets))
        return deduped, "; ".join(reasons)
