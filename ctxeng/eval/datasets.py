from __future__ import annotations

from dataclasses import dataclass

from ctxeng.models import MemoryItem


@dataclass
class EvalQuery:
    user_id: str
    query: str
    relevant_ids: set[str]


@dataclass
class EvalDataset:
    name: str
    description: str
    memories: list[MemoryItem]
    queries: list[EvalQuery]


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
        EvalQuery(
            user_id="alice", query="What do you do for fun?", relevant_ids={"m1", "m3", "m4"}
        ),
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
        EvalQuery(
            user_id="alice", query="What programming language do I like?", relevant_ids={"m1", "m2"}
        ),
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


def _build_memory_retention() -> EvalDataset:
    memories = [
        MemoryItem(user_id="charlie", text="My favorite book is Dune.", id="m1"),
        MemoryItem(user_id="charlie", text="I have a dog named Max.", id="m2"),
        MemoryItem(user_id="charlie", text="I work as a data scientist.", id="m3"),
        MemoryItem(user_id="charlie", text="I visited Japan last year.", id="m4"),
        MemoryItem(user_id="charlie", text="I enjoy playing the guitar.", id="m5"),
        MemoryItem(user_id="charlie", text="My team uses Kubernetes.", id="m6"),
        MemoryItem(user_id="charlie", text="I am learning Mandarin Chinese.", id="m7"),
    ]
    queries = [
        EvalQuery(user_id="charlie", query="What is my favorite book?", relevant_ids={"m1"}),
        EvalQuery(user_id="charlie", query="What pet do I have?", relevant_ids={"m2"}),
        EvalQuery(user_id="charlie", query="What is my job?", relevant_ids={"m3"}),
        EvalQuery(user_id="charlie", query="Where did I travel recently?", relevant_ids={"m4"}),
        EvalQuery(user_id="charlie", query="What instrument do I play?", relevant_ids={"m5"}),
        EvalQuery(
            user_id="charlie",
            query="What orchestration tool does my team use?",
            relevant_ids={"m6"},
        ),
        EvalQuery(user_id="charlie", query="What language am I studying?", relevant_ids={"m7"}),
    ]
    return EvalDataset(
        name="memory_retention",
        description="7 distinct personal facts to test memory recall across turns",
        memories=memories,
        queries=queries,
    )


def _build_long_context_coherence() -> EvalDataset:
    memories = [
        MemoryItem(user_id="diana", text="The company mission is to democratize AI.", id="m1"),
        MemoryItem(user_id="diana", text="Our product roadmap has 3 phases.", id="m2"),
        MemoryItem(user_id="diana", text="Phase 1 focuses on data collection pipeline.", id="m3"),
        MemoryItem(user_id="diana", text="Phase 2 adds model training infrastructure.", id="m4"),
        MemoryItem(user_id="diana", text="Phase 3 is about deployment and monitoring.", id="m5"),
        MemoryItem(
            user_id="diana", text="We use Apache Kafka for real-time data streaming.", id="m6"
        ),
        MemoryItem(user_id="diana", text="Models are trained on A100 GPUs.", id="m7"),
        MemoryItem(user_id="diana", text="We have a team of 15 ML engineers.", id="m8"),
        MemoryItem(
            user_id="diana", text="The stack includes PyTorch, Ray, and Kubernetes.", id="m9"
        ),
        MemoryItem(user_id="diana", text="We deploy models via Triton Inference Server.", id="m10"),
        MemoryItem(user_id="diana", text="Monitoring uses Prometheus and Grafana.", id="m11"),
        MemoryItem(user_id="diana", text="Our customers are in healthcare and finance.", id="m12"),
        MemoryItem(user_id="diana", text="We process about 10TB of data daily.", id="m13"),
        MemoryItem(user_id="diana", text="The main office is in San Francisco.", id="m14"),
        MemoryItem(user_id="diana", text="We have remote workers in 8 time zones.", id="m15"),
    ]
    queries = [
        EvalQuery(
            user_id="diana",
            query="What is the company mission and roadmap?",
            relevant_ids={"m1", "m2", "m3", "m4", "m5"},
        ),
        EvalQuery(
            user_id="diana",
            query="What ML infrastructure do we use?",
            relevant_ids={"m6", "m7", "m9", "m10", "m11"},
        ),
        EvalQuery(
            user_id="diana",
            query="Tell me about the team and operations",
            relevant_ids={"m8", "m12", "m13", "m14", "m15"},
        ),
        EvalQuery(
            user_id="diana",
            query="What is the data pipeline and training setup?",
            relevant_ids={"m3", "m6", "m7", "m9", "m13"},
        ),
        EvalQuery(
            user_id="diana",
            query="How is deployment and monitoring handled?",
            relevant_ids={"m5", "m10", "m11"},
        ),
    ]
    return EvalDataset(
        name="long_context_coherence",
        description="15 cross-referencing facts about a company to test coherence across a long context window",
        memories=memories,
        queries=queries,
    )


BUILT_IN_DATASETS: dict[str, EvalDataset] = {
    "simple_preferences": _build_simple_preferences(),
    "multi_topic_conversations": _build_multi_topic(),
    "long_term_cross_session": _build_cross_session(),
    "memory_retention": _build_memory_retention(),
    "long_context_coherence": _build_long_context_coherence(),
}


def get_dataset(name: str) -> EvalDataset:
    if name in BUILT_IN_DATASETS:
        return BUILT_IN_DATASETS[name]
    raise KeyError(f"Unknown dataset: {name}. Available: {list(BUILT_IN_DATASETS.keys())}")


def list_datasets() -> list[str]:
    return list(BUILT_IN_DATASETS.keys())
