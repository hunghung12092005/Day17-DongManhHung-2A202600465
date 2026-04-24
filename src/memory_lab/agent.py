from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from memory_lab.backends import MemoryStack
from memory_lab.config import Settings
from memory_lab.context import ContextWindowManager
from memory_lab.models import AgentState, MemorySource, RetrievedItem
from memory_lab.router import MemoryRouter


SYSTEM_PROMPT = """You are a memory-aware AI assistant.
Use retrieved memory when it is relevant.
If memory is missing, say so briefly instead of hallucinating.
Prefer concise Vietnamese answers with specific details."""


class ResponseGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.mode = settings.memory_lab_mode.lower()
        self.llm = None
        if self.mode != "offline" and settings.openai_api_key:
            self.llm = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.2,
                max_tokens=settings.max_response_tokens,
            )

    def generate(self, user_input: str, context: str, used_sources: list[MemorySource]) -> str:
        if self.llm is not None:
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Retrieved context:\n{context}\n\n"
                f"User question: {user_input}\n"
                "Answer in Vietnamese."
            )
            return self.llm.invoke(prompt).content
        return self._offline_answer(user_input, context, used_sources)

    def _offline_answer(self, user_input: str, context: str, used_sources: list[MemorySource]) -> str:
        lower = user_input.lower()
        sections: dict[str, list[str]] = {"system": [], "history": [], "memory": []}
        current_label = "system"
        for raw_line in context.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line == "[SYSTEM]":
                current_label = "system"
                continue
            if line == "[HISTORY]":
                current_label = "history"
                continue
            if line == "[MEMORY]":
                current_label = "memory"
                continue
            sections.setdefault(current_label, []).append(line)

        memory_text = " ".join(sections.get("memory", []))
        history_text = " ".join(sections.get("history", []))
        joined = memory_text or history_text
        if "semantic" in used_sources and ("là gì" in lower or "giải thích" in lower or "so sánh" in lower):
            return f"Dựa trên semantic memory: {joined[:420]}"
        if "long_term" in used_sources and any(
            word in lower for word in ("thích", "sở thích", "ưu tiên", "nhớ tôi", "hồ sơ", "mục tiêu")
        ):
            return f"Theo long-term memory của bạn: {joined[:420]}"
        if "episodic" in used_sources and any(word in lower for word in ("lần trước", "kinh nghiệm", "sự cố", "bài học")):
            return f"Dựa trên episodic memory: {joined[:420]}"
        if joined:
            return f"Tôi tìm thấy ngữ cảnh liên quan: {joined[:420]}"
        return "Hiện chưa có đủ memory liên quan trong context để trả lời chắc chắn."


@dataclass
class MemoryAgent:
    settings: Settings
    memories: MemoryStack
    router: MemoryRouter
    context_manager: ContextWindowManager
    generator: ResponseGenerator

    def build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("route_node", self._route)
        graph.add_node("retrieve_node", self._retrieve)
        graph.add_node("context_node", self._build_context)
        graph.add_node("answer_node", self._answer)
        graph.add_node("persist_node", self._persist)
        graph.add_edge(START, "route_node")
        graph.add_edge("route_node", "retrieve_node")
        graph.add_edge("retrieve_node", "context_node")
        graph.add_edge("context_node", "answer_node")
        graph.add_edge("answer_node", "persist_node")
        graph.add_edge("persist_node", END)
        return graph.compile()

    def ask(self, session_id: str, user_input: str) -> AgentState:
        self.memories.seed_defaults(session_id)
        app = self.build_graph()
        return app.invoke({"session_id": session_id, "user_input": user_input})

    def seed_semantic_memory(self, path: Path) -> None:
        docs = json.loads(path.read_text(encoding="utf-8"))
        self.memories.semantic.seed_documents(docs)

    def _route(self, state: AgentState) -> AgentState:
        targets, reason = self.router.route(state["user_input"])
        return {"route_targets": targets, "route_reason": reason}

    def _retrieve(self, state: AgentState) -> AgentState:
        query = state["user_input"]
        session_id = state["session_id"]
        targets = state["route_targets"]
        retrieved: list[RetrievedItem] = []
        history_items = self.memories.short_term.search(session_id, query, limit=6)
        short_term_top = history_items[:3]
        retrieved.extend(short_term_top)

        if "long_term" in targets:
            retrieved.extend(self.memories.long_term.search(session_id, query))
        if "episodic" in targets:
            retrieved.extend(self.memories.episodic.search(query))
        if "semantic" in targets:
            retrieved.extend(self.memories.semantic.search(query))

        used_sources = list(dict.fromkeys(item["source"] for item in retrieved))
        memory_hits = {source: source in used_sources for source in targets}
        return {
            "retrieved_items": retrieved,
            "used_sources": used_sources,
            "memory_hits": memory_hits,
        }

    def _build_context(self, state: AgentState) -> AgentState:
        history = [item for item in state["retrieved_items"] if item["source"] == "short_term"]
        memories = [item for item in state["retrieved_items"] if item["source"] != "short_term"]
        prompt_context, token_breakdown = self.context_manager.build_context(
            system_prompt=SYSTEM_PROMPT,
            history_items=history,
            memory_items=memories,
            user_input=state["user_input"],
        )
        return {"prompt_context": prompt_context, "token_breakdown": token_breakdown}

    def _answer(self, state: AgentState) -> AgentState:
        answer = self.generator.generate(
            user_input=state["user_input"],
            context=state["prompt_context"],
            used_sources=state.get("used_sources", []),
        )
        token_breakdown = dict(state["token_breakdown"])
        token_breakdown["response_tokens"] = self.context_manager.count_tokens(answer)
        token_breakdown["total_tokens"] = (
            token_breakdown["total_prompt_tokens"] + token_breakdown["response_tokens"]
        )
        return {"answer": answer, "token_breakdown": token_breakdown}

    def _persist(self, state: AgentState) -> AgentState:
        user_input = state["user_input"]
        answer = state["answer"]
        self.memories.short_term.save_turn(state["session_id"], user_input, answer)

        inferred_facts = self._extract_profile_facts(user_input)
        for key, value in inferred_facts.items():
            self.memories.long_term.write_fact(state["session_id"], key, value)

        if any(word in user_input.lower() for word in ("lần trước", "kinh nghiệm", "sự cố", "bài học")):
            self.memories.episodic.append_episode(
                summary=user_input,
                tags=["user_query", "reflection"],
                outcome=answer[:220],
            )
        return state

    def _extract_profile_facts(self, user_input: str) -> dict[str, str]:
        text = user_input.lower()
        patterns = {
            "ngôn ngữ yêu thích": r"tôi thích ([a-zA-Z0-9+#\-\s]+)",
            "phong cách trả lời": r"hãy trả lời ([a-zA-Z0-9à-ỹ\s,]+)",
        }
        facts: dict[str, str] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                facts[key] = match.group(1).strip()
        return facts


class BaselineAgent:
    def __init__(self, settings: Settings) -> None:
        self.context_manager = ContextWindowManager(
            model_name=settings.openai_model,
            token_budget=settings.token_budget,
            max_response_tokens=settings.max_response_tokens,
        )
        self.generator = ResponseGenerator(settings)

    def ask(self, session_id: str, user_input: str) -> AgentState:
        prompt_context, token_breakdown = self.context_manager.build_context(
            system_prompt=SYSTEM_PROMPT,
            history_items=[],
            memory_items=[],
            user_input=user_input,
        )
        answer = self.generator.generate(user_input=user_input, context=prompt_context, used_sources=[])
        token_breakdown["response_tokens"] = self.context_manager.count_tokens(answer)
        token_breakdown["total_tokens"] = (
            token_breakdown["total_prompt_tokens"] + token_breakdown["response_tokens"]
        )
        return {
            "session_id": session_id,
            "user_input": user_input,
            "answer": answer,
            "route_targets": [],
            "retrieved_items": [],
            "used_sources": [],
            "memory_hits": {},
            "token_breakdown": token_breakdown,
            "prompt_context": prompt_context,
        }
