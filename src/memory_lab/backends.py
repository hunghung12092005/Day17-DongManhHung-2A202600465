from __future__ import annotations

import hashlib
import json
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Iterable

import numpy as np
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage

from memory_lab.models import RetrievedItem

try:
    import opentelemetry.sdk._logs as otel_logs
    import opentelemetry.sdk._logs.export as otel_logs_export

    if not hasattr(otel_logs, "ReadableLogRecord") and hasattr(otel_logs, "LogData"):
        otel_logs.ReadableLogRecord = otel_logs.LogData
    if not hasattr(otel_logs_export, "LogRecordExportResult") and hasattr(
        otel_logs_export, "LogExportResult"
    ):
        otel_logs_export.LogRecordExportResult = otel_logs_export.LogExportResult
except Exception:
    pass

import chromadb


def _normalize_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", text.lower()) if len(token) > 2}


def _overlap_score(left: str, right: str) -> float:
    a = _normalize_tokens(left)
    b = _normalize_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ShortTermMemoryBackend:
    def __init__(self) -> None:
        self.memories: dict[str, ConversationBufferMemory] = {}

    def _get_memory(self, session_id: str) -> ConversationBufferMemory:
        if session_id not in self.memories:
            self.memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="input",
                output_key="output",
                return_messages=True,
            )
        return self.memories[session_id]

    def save_turn(self, session_id: str, user_input: str, answer: str) -> None:
        self._get_memory(session_id).save_context({"input": user_input}, {"output": answer})

    def get_messages(self, session_id: str) -> list:
        variables = self._get_memory(session_id).load_memory_variables({})
        return list(variables.get("chat_history", []))

    def search(self, session_id: str, query: str, limit: int = 6) -> list[RetrievedItem]:
        items: list[RetrievedItem] = []
        for message in self.get_messages(session_id):
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            score = _overlap_score(query, message.content)
            if score == 0:
                continue
            items.append(
                {
                    "source": "short_term",
                    "content": f"{role}: {message.content}",
                    "score": score,
                    "priority": 2 if role == "assistant" else 1,
                    "metadata": {"role": role},
                }
            )
        items.sort(key=lambda item: item["score"], reverse=True)
        return items[:limit]


class RedisLongTermMemoryBackend:
    def __init__(self, redis_url: str, namespace: str = "memory-lab") -> None:
        self.namespace = namespace
        parsed = urlparse(redis_url)
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or 6379
        self.db = int((parsed.path or "/0").lstrip("/"))

    def _profile_key(self, session_id: str) -> str:
        return f"{self.namespace}:profile:{session_id}"

    def _encode_command(self, *parts: str) -> bytes:
        payload = [f"*{len(parts)}\r\n".encode("utf-8")]
        for part in parts:
            encoded = str(part).encode("utf-8")
            payload.append(f"${len(encoded)}\r\n".encode("utf-8"))
            payload.append(encoded + b"\r\n")
        return b"".join(payload)

    def _read_line(self, conn: socket.socket) -> bytes:
        data = bytearray()
        while not data.endswith(b"\r\n"):
            chunk = conn.recv(1)
            if not chunk:
                raise RuntimeError("Redis connection closed unexpectedly.")
            data.extend(chunk)
        return bytes(data[:-2])

    def _read_resp(self, conn: socket.socket):
        prefix = conn.recv(1)
        if not prefix:
            raise RuntimeError("Redis connection closed unexpectedly.")
        if prefix == b"+":
            return self._read_line(conn).decode("utf-8")
        if prefix == b":":
            return int(self._read_line(conn))
        if prefix == b"$":
            length = int(self._read_line(conn))
            if length == -1:
                return None
            data = bytearray()
            while len(data) < length + 2:
                chunk = conn.recv(length + 2 - len(data))
                if not chunk:
                    raise RuntimeError("Redis connection closed unexpectedly.")
                data.extend(chunk)
            return bytes(data[:-2]).decode("utf-8")
        if prefix == b"*":
            length = int(self._read_line(conn))
            return [self._read_resp(conn) for _ in range(length)]
        if prefix == b"-":
            raise RuntimeError(self._read_line(conn).decode("utf-8"))
        raise RuntimeError(f"Unsupported Redis RESP prefix: {prefix!r}")

    def _command(self, *parts: str):
        with socket.create_connection((self.host, self.port), timeout=2) as conn:
            if self.db:
                conn.sendall(self._encode_command("SELECT", str(self.db)))
                self._read_resp(conn)
            conn.sendall(self._encode_command(*parts))
            return self._read_resp(conn)

    def write_fact(self, session_id: str, key: str, value: str) -> None:
        self._command("HSET", self._profile_key(session_id), key, value)

    def get_all_facts(self, session_id: str) -> dict[str, str]:
        result = self._command("HGETALL", self._profile_key(session_id))
        if not result:
            return {}
        return dict(zip(result[::2], result[1::2]))

    def search(self, session_id: str, query: str, limit: int = 5) -> list[RetrievedItem]:
        facts = self.get_all_facts(session_id)
        results: list[RetrievedItem] = []
        for key, value in facts.items():
            content = f"{key}: {value}"
            score = _overlap_score(query, content)
            if score == 0 and key not in query.lower():
                continue
            results.append(
                {
                    "source": "long_term",
                    "content": content,
                    "score": max(score, 0.2),
                    "priority": 1,
                    "metadata": {"key": key},
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]


class EpisodicMemoryBackend:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _load(self) -> list[dict]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, episodes: list[dict]) -> None:
        self.path.write_text(json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_episode(self, summary: str, tags: list[str], outcome: str) -> None:
        episodes = self._load()
        episodes.append({"summary": summary, "tags": tags, "outcome": outcome})
        self._save(episodes)

    def search(self, query: str, limit: int = 4) -> list[RetrievedItem]:
        results: list[RetrievedItem] = []
        for episode in self._load():
            content = f"{episode['summary']} Outcome: {episode['outcome']}"
            tags = " ".join(episode.get("tags", []))
            score = max(_overlap_score(query, content), _overlap_score(query, tags))
            if score == 0:
                continue
            results.append(
                {
                    "source": "episodic",
                    "content": content,
                    "score": score,
                    "priority": 2,
                    "metadata": {"tags": episode.get("tags", [])},
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]


class HashEmbeddings:
    def __init__(self, dims: int = 128) -> None:
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dims, dtype=np.float32)
        for token in _normalize_tokens(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest, 16) % self.dims
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        if norm:
            vector = vector / norm
        return vector.tolist()


class SemanticMemoryBackend:
    def __init__(self, persist_dir: Path, collection_name: str = "semantic_memory") -> None:
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embeddings = HashEmbeddings()

    def seed_documents(self, docs: Iterable[dict]) -> None:
        docs = list(docs)
        if not docs:
            return
        existing_ids = set(self.collection.get(include=[])["ids"])
        to_add = [doc for doc in docs if doc["id"] not in existing_ids]
        if not to_add:
            return
        self.collection.add(
            ids=[doc["id"] for doc in to_add],
            documents=[doc["text"] for doc in to_add],
            metadatas=[doc.get("metadata", {}) for doc in to_add],
            embeddings=[self.embeddings.embed(doc["text"]) for doc in to_add],
        )

    def search(self, query: str, limit: int = 4) -> list[RetrievedItem]:
        result = self.collection.query(
            query_embeddings=[self.embeddings.embed(query)],
            n_results=limit,
        )
        items: list[RetrievedItem] = []
        distances = result.get("distances", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        for distance, document, metadata in zip(distances, docs, metadatas):
            score = 1 / (1 + float(distance))
            items.append(
                {
                    "source": "semantic",
                    "content": document,
                    "score": score,
                    "priority": 2,
                    "metadata": metadata or {},
                }
            )
        return items


@dataclass
class MemoryStack:
    short_term: ShortTermMemoryBackend
    long_term: RedisLongTermMemoryBackend
    episodic: EpisodicMemoryBackend
    semantic: SemanticMemoryBackend

    def seed_defaults(self, session_id: str) -> None:
        default_profile = {
            "ngôn ngữ yêu thích": "Python",
            "phong cách trả lời": "ngắn gọn, có ví dụ thực tế",
            "mục tiêu dự án": "xây agent có memory trên LangGraph",
        }
        for key, value in default_profile.items():
            self.long_term.write_fact(session_id, key, value)

        if not self.episodic.search("redis lỗi benchmark"):
            self.episodic.append_episode(
                summary="Trong lần benchmark trước, agent gặp lỗi khi Redis chưa khởi động.",
                tags=["redis", "benchmark", "setup"],
                outcome="Cần kiểm tra Redis health trước khi chạy benchmark.",
            )
            self.episodic.append_episode(
                summary="Agent từng trả lời tốt hơn khi ưu tiên semantic memory cho câu hỏi kiến thức.",
                tags=["semantic", "routing", "knowledge"],
                outcome="Nên route câu hỏi factual sang semantic memory trước.",
            )
