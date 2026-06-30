from __future__ import annotations

from ctxeng.llm.base import LLMMessage, LLMProvider, LLMResponse


class AlwaysEchoProvider(LLMProvider):
    @property
    def model_name(self) -> str:
        return "echo"

    def generate(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        return LLMResponse(
            content=messages[-1].content if messages else "",
            finish_reason="stop",
        )

    def stream(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        return self.generate(messages)


class TestLLMProvider:
    def test_provider_returns_response(self) -> None:
        provider = AlwaysEchoProvider()
        resp = provider.generate(
            [
                LLMMessage(role="user", content="hello"),
            ]
        )
        assert resp.content == "hello"
        assert resp.finish_reason == "stop"

    def test_stream_returns_response(self) -> None:
        provider = AlwaysEchoProvider()
        resp = provider.stream(
            [
                LLMMessage(role="user", content="test stream"),
            ]
        )
        assert resp.content == "test stream"

    def test_model_name(self) -> None:
        provider = AlwaysEchoProvider()
        assert provider.model_name == "echo"

    def test_multiple_messages(self) -> None:
        provider = AlwaysEchoProvider()
        resp = provider.generate(
            [
                LLMMessage(role="system", content="be helpful"),
                LLMMessage(role="user", content="what's up"),
            ]
        )
        assert resp.content == "what's up"


class TestLLMMessage:
    def test_create_system_message(self) -> None:
        msg = LLMMessage(role="system", content="be concise")
        assert msg.role == "system"
        assert msg.content == "be concise"

    def test_create_user_message(self) -> None:
        msg = LLMMessage(role="user", content="hello")
        assert msg.role == "user"


class TestGenerateReply:
    def test_generate_reply(self) -> None:
        from ctxeng.llm.chat import generate_reply

        provider = AlwaysEchoProvider()
        resp = generate_reply(provider, "system prompt", "user query")
        assert resp.content == "user query"
