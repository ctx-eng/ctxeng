from fastapi.testclient import TestClient

from ctxeng.server import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_add_and_search_memory():
    r = client.post("/memories", json={"user_id": "bob", "text": "Bob likes details."})
    assert r.status_code == 201
    data = r.json()
    assert data["user_id"] == "bob"
    assert data["text"] == "Bob likes details."

    r = client.get("/memories/bob?query=details")
    assert r.status_code == 200
    results = r.json()
    assert len(results) == 1
    assert results[0]["text"] == "Bob likes details."


def test_build_prompt():
    body = {
        "user_id": "bob",
        "turns": [
            {"role": "user", "content": "What's the capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ],
        "current_query": "Tell me more",
    }
    r = client.post("/prompt", json=body)
    assert r.status_code == 200
    prompt = r.json()["prompt"]
    assert "What's the capital of France?" in prompt
    assert "Paris." in prompt
    assert "Tell me more" in prompt
