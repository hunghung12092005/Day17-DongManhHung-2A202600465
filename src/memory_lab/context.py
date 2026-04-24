from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from memory_lab.models import RetrievedItem


@dataclass
class ContextChunk:
    label: str
    content: str
    priority: int


class ContextWindowManager:
    def __init__(self, model_name: str, token_budget: int, max_response_tokens: int) -> None:
        self.model_name = model_name
        self.token_budget = token_budget
        self.max_response_tokens = max_response_tokens
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def build_context(
        self,
        system_prompt: str,
        history_items: list[RetrievedItem],
        memory_items: list[RetrievedItem],
        user_input: str,
    ) -> tuple[str, dict[str, int]]:
        chunks = [ContextChunk("system", system_prompt, 1)]
        chunks.extend(ContextChunk("history", item["content"], 3) for item in history_items)
        chunks.extend(ContextChunk("memory", item["content"], item["priority"]) for item in memory_items)
        chunks.append(ContextChunk("user", user_input, 1))

        reserve = self.max_response_tokens
        current_tokens = sum(self.count_tokens(chunk.content) for chunk in chunks)

        while current_tokens + reserve > self.token_budget:
            removable = [chunk for chunk in chunks if chunk.label in {"history", "memory"}]
            if not removable:
                break
            removable.sort(key=lambda chunk: (chunk.priority, self.count_tokens(chunk.content)), reverse=True)
            victim = removable[0]
            chunks.remove(victim)
            current_tokens = sum(self.count_tokens(chunk.content) for chunk in chunks)

        prompt_context = "\n\n".join(
            f"[{chunk.label.upper()}]\n{chunk.content}" for chunk in chunks if chunk.label != "user"
        )
        breakdown = {
            "system_tokens": sum(self.count_tokens(chunk.content) for chunk in chunks if chunk.label == "system"),
            "history_tokens": sum(self.count_tokens(chunk.content) for chunk in chunks if chunk.label == "history"),
            "memory_tokens": sum(self.count_tokens(chunk.content) for chunk in chunks if chunk.label == "memory"),
            "user_tokens": sum(self.count_tokens(chunk.content) for chunk in chunks if chunk.label == "user"),
            "reserved_response_tokens": reserve,
            "total_prompt_tokens": current_tokens,
        }
        return prompt_context, breakdown
