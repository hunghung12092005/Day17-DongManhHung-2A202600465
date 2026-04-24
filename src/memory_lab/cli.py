from __future__ import annotations

import argparse
import json

from memory_lab.agent import BaselineAgent, MemoryAgent, ResponseGenerator
from memory_lab.backends import (
    EpisodicMemoryBackend,
    MemoryStack,
    RedisLongTermMemoryBackend,
    SemanticMemoryBackend,
    ShortTermMemoryBackend,
)
from memory_lab.benchmark import BenchmarkRunner, render_markdown_report
from memory_lab.config import get_settings
from memory_lab.context import ContextWindowManager
from memory_lab.router import MemoryRouter


def build_memory_agent() -> MemoryAgent:
    settings = get_settings()
    memories = MemoryStack(
        short_term=ShortTermMemoryBackend(),
        long_term=RedisLongTermMemoryBackend(settings.redis_url),
        episodic=EpisodicMemoryBackend(settings.episodic_path),
        semantic=SemanticMemoryBackend(settings.chroma_dir),
    )
    memories.seed_defaults("demo-user")
    agent = MemoryAgent(
        settings=settings,
        memories=memories,
        router=MemoryRouter(),
        context_manager=ContextWindowManager(
            model_name=settings.openai_model,
            token_budget=settings.token_budget,
            max_response_tokens=settings.max_response_tokens,
        ),
        generator=ResponseGenerator(settings),
    )
    agent.seed_semantic_memory(settings.semantic_seed_path)
    return agent


def run_benchmark() -> None:
    settings = get_settings()
    agent = build_memory_agent()
    baseline = BaselineAgent(settings)
    runner = BenchmarkRunner(agent, baseline)
    report = runner.run(settings.benchmark_path)
    markdown = render_markdown_report(report)
    settings.report_path.write_text(markdown, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nMarkdown report written to: {settings.report_path}")


def run_demo() -> None:
    agent = build_memory_agent()
    session_id = "demo-user"
    print("Memory lab demo. Type 'exit' to quit.")
    while True:
        user_input = input("Bạn: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        state = agent.ask(session_id, user_input)
        print(f"Agent: {state['answer']}")
        print(f"Route: {state['route_targets']}")
        print(f"Used sources: {state.get('used_sources', [])}")
        print(f"Token breakdown: {state['token_breakdown']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph memory lab")
    parser.add_argument("command", choices=["benchmark", "demo"])
    args = parser.parse_args()

    if args.command == "benchmark":
        run_benchmark()
    elif args.command == "demo":
        run_demo()


if __name__ == "__main__":
    main()
