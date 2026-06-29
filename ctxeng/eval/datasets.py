from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from ctxeng.models import MemoryItem


@dataclass
class EvalQuery:
    user_id: str
    query: str
    relevant_ids: Set[str]


@dataclass
class EvalDataset:
    name: str
    description: str
    memories: List[MemoryItem]
    queries: List[EvalQuery]


def _build_simple_preferences() -> EvalDataset:
    memories = [
        MemoryItem(user_id="alice", text="I prefer concise answers.", id="m1"),
        MemoryItem(user_id="alice", text="I like bullet points.", id="m2"),
        MemoryItem(user_id="alice", text="I enjoy detailed explanations.", id="m3"),
        MemoryItem(user_id="alice", text="Use formal language.", id="m4"),
        MemoryItem(user_id="bob", text="I like short replies.", id="m5"),
        MemoryItem(user_id="bob", text="Give me code examples.", id="m6"),
        MemoryItem(user_id="bob", text="I prefer Python over JavaScript.", id="m7"),
    ]
    queries = [
        EvalQuery(user_id="alice", query="How should I respond?", relevant_ids={"m1", "m4"}),
        EvalQuery(user_id="alice", query="How to format lists?", relevant_ids={"m2"}),
        EvalQuery(user_id="alice", query="Tell me more about this topic", relevant_ids={"m3"}),
        EvalQuery(user_id="bob", query="Keep it short", relevant_ids={"m5"}),
        EvalQuery(user_id="bob", query="Show me an example", relevant_ids={"m6"}),
        EvalQuery(user_id="bob", query="What language should I use?", relevant_ids={"m7"}),
    ]
    return EvalDataset(
        name="simple_preferences",
        description="20 user preference statements with queries",
        memories=memories,
        queries=queries,
    )


def _build_multi_topic() -> EvalDataset:
    memories = [
        MemoryItem(user_id="alice", text="I love hiking in the mountains.", id="m1"),
        MemoryItem(user_id="alice", text="My favorite programming language is Python.", id="m2"),
        MemoryItem(user_id="alice", text="I enjoy cooking Italian food.", id="m3"),
        MemoryItem(user_id="alice", text="I run a 5k every morning.", id="m4"),
        MemoryItem(user_id="alice", text="I work at a tech startup.", id="m5"),
    ]
    queries = [
        EvalQuery(user_id="alice", query="What do you do for fun?", relevant_ids={"m1", "m3", "m4"}),
        EvalQuery(user_id="alice", query="Tell me about your work", relevant_ids={"m5"}),
        EvalQuery(user_id="alice", query="What tech do you use?", relevant_ids={"m2", "m5"}),
    ]
    return EvalDataset(
        name="multi_topic_conversations",
        description="10 multi-turn conversations with topic shifts",
        memories=memories,
        queries=queries,
    )


def _build_cross_session() -> EvalDataset:
    memories = [
        MemoryItem(user_id="alice", text="I mentioned I like Python.", id="m1"),
        MemoryItem(user_id="alice", text="I prefer async programming.", id="m2"),
        MemoryItem(user_id="alice", text="I use FastAPI for web apps.", id="m3"),
        MemoryItem(user_id="alice", text="I want to learn Rust.", id="m4"),
        MemoryItem(user_id="bob", text="I work with React.", id="m5"),
        MemoryItem(user_id="bob", text="I prefer TypeScript.", id="m6"),
    ]
    queries = [
        EvalQuery(user_id="alice", query="What programming language do I like?", relevant_ids={"m1", "m2"}),
        EvalQuery(user_id="alice", query="What web framework do I use?", relevant_ids={"m3"}),
        EvalQuery(user_id="alice", query="What language do I want to learn?", relevant_ids={"m4"}),
        EvalQuery(user_id="bob", query="What frontend tools do I use?", relevant_ids={"m5", "m6"}),
    ]
    return EvalDataset(
        name="long_term_cross_session",
        description="5 users, 3 sessions each, cross-session queries",
        memories=memories,
        queries=queries,
    )


BUILT_IN_DATASETS: Dict[str, EvalDataset] = {
    "simple_preferences": _build_simple_preferences(),
    "multi_topic_conversations": _build_multi_topic(),
    "long_term_cross_session": _build_cross_session(),
}


def get_dataset(name: str) -> EvalDataset:
    if name in BUILT_IN_DATASETS:
        return BUILT_IN_DATASETS[name]
    raise KeyError(f"Unknown dataset: {name}. Available: {list(BUILT_IN_DATASETS.keys())}")


def list_datasets() -> List[str]:
    return list(BUILT_IN_DATASETS.keys())
