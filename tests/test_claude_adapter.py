from types import SimpleNamespace
from typing import Any, cast

import pytest
from anthropic import AsyncAnthropic, omit
from anthropic.types.beta import BetaMessage

from cerebro import (
    Message,
    ModelRequest,
    ProviderBlock,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
from cerebro.providers.claude import (
    FALLBACK_BETA,
    ClaudeClient,
    from_anthropic_message,
    to_anthropic_messages,
    to_anthropic_tool,
)

THINKING = {"type": "thinking", "thinking": "Add them.", "signature": "sig123"}
THINKING_BLOCK = ProviderBlock(
    provider="anthropic", kind="thinking", text="Add them.", data=THINKING
)
FALLBACK = {
    "type": "fallback",
    "from": {"model": "claude-opus-5-5"},
    "to": {"model": "claude-opus-4-8"},
    "trigger": {"type": "refusal", "category": "cyber"},
}


def api_message(content: list[dict[str, Any]], stop_reason: str = "end_turn") -> BetaMessage:
    return BetaMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )


# --- core -> Anthropic ---


def test_messages_translate_to_anthropic_blocks() -> None:
    messages = [
        Message.user("2+3?"),
        Message(
            role="assistant",
            content=[
                THINKING_BLOCK,
                TextBlock(text="Let me add."),
                ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3}),
            ],
        ),
        Message(role="user", content=[ToolResult(call_id="c1", content="boom", is_error=True)]),
    ]
    assert to_anthropic_messages(messages) == [
        {"role": "user", "content": [{"type": "text", "text": "2+3?"}]},
        {
            "role": "assistant",
            "content": [
                THINKING,
                {"type": "text", "text": "Let me add."},
                {"type": "tool_use", "id": "c1", "name": "add", "input": {"a": 2, "b": 3}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": "boom", "is_error": True}
            ],
        },
    ]


def test_other_providers_blocks_are_skipped() -> None:
    message = Message(
        role="assistant",
        content=[
            ProviderBlock(provider="openai", kind="reasoning", data={"id": "r1"}),
            TextBlock(text="Hi"),
        ],
    )
    assert to_anthropic_messages([message]) == [
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]}
    ]


def test_tool_spec_translates() -> None:
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    spec = ToolSpec(name="search", description="Search notes.", input_schema=schema)
    assert to_anthropic_tool(spec) == {
        "name": "search",
        "description": "Search notes.",
        "input_schema": schema,
    }


# --- Anthropic -> core ---


def test_response_translates_to_core_types() -> None:
    message = api_message(
        [
            THINKING,
            {"type": "text", "text": "Let me add."},
            {"type": "tool_use", "id": "c1", "name": "add", "input": {"a": 2, "b": 3}},
        ],
        stop_reason="tool_use",
    )
    response = from_anthropic_message(message)

    assert response.stop_reason == "tool_use"
    assert response.usage == Usage(input_tokens=10, output_tokens=5)
    assert response.message.content == [
        THINKING_BLOCK,
        TextBlock(text="Let me add."),
        ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3}),
    ]


def test_unknown_blocks_round_trip_unchanged() -> None:
    # The fallback block's "from" field is "from_" in Python; it must go back as "from".
    response = from_anthropic_message(api_message([FALLBACK, {"type": "text", "text": "Hi"}]))
    block = response.message.content[0]
    assert isinstance(block, ProviderBlock)
    assert block.kind == "fallback"

    assert to_anthropic_messages([response.message]) == [
        {"role": "assistant", "content": [FALLBACK, {"type": "text", "text": "Hi"}]}
    ]


@pytest.mark.parametrize(
    ("api_reason", "expected"),
    [
        ("end_turn", "end_turn"),
        ("stop_sequence", "end_turn"),
        ("tool_use", "tool_use"),
        ("max_tokens", "max_tokens"),
        ("model_context_window_exceeded", "max_tokens"),
        ("refusal", "refusal"),
        ("pause_turn", "other"),
    ],
)
def test_stop_reasons_map(api_reason: str, expected: str) -> None:
    message = api_message([{"type": "text", "text": "x"}], stop_reason=api_reason)
    assert from_anthropic_message(message).stop_reason == expected


# --- the client call itself ---


class StubMessages:
    """Stands in for `AsyncAnthropic().beta.messages`; records the call."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> BetaMessage:
        self.kwargs = kwargs
        return api_message([{"type": "text", "text": "Hello!"}])


def stub_client() -> tuple[AsyncAnthropic, StubMessages]:
    messages = StubMessages()
    return cast(AsyncAnthropic, SimpleNamespace(beta=SimpleNamespace(messages=messages))), messages


async def test_complete_sends_request_with_fallbacks() -> None:
    sdk, stub = stub_client()
    request = ModelRequest(
        model="claude-opus-5-5", system="Be brief.", messages=[Message.user("Hi")], max_tokens=100
    )
    response = await ClaudeClient(client=sdk).complete(request)

    assert response.message.text == "Hello!"
    assert stub.kwargs["model"] == "claude-opus-5-5"
    assert stub.kwargs["system"] == "Be brief."
    assert stub.kwargs["max_tokens"] == 100
    assert stub.kwargs["tools"] is omit  # no tools: leave the field out entirely
    assert stub.kwargs["betas"] == [FALLBACK_BETA]
    assert stub.kwargs["fallbacks"] == "default"


async def test_complete_can_disable_fallbacks_and_omits_empty_system() -> None:
    sdk, stub = stub_client()
    request = ModelRequest(model="m", system="", messages=[Message.user("Hi")], max_tokens=100)
    await ClaudeClient(client=sdk, fallbacks=False).complete(request)

    assert stub.kwargs["system"] is omit
    assert stub.kwargs["betas"] is omit
    assert stub.kwargs["fallbacks"] is omit
