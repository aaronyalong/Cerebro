import pytest
from pydantic import ValidationError

from cerebro import Message, TextBlock, ToolCall, ToolResult, Usage


def test_user_helper_wraps_text() -> None:
    message = Message.user("hi")
    assert message.role == "user"
    assert message.content == [TextBlock(text="hi")]


def test_text_and_tool_calls_helpers() -> None:
    call = ToolCall(id="c1", name="search", arguments={"q": "x"})
    message = Message(
        role="assistant",
        content=[TextBlock(text="Let me "), call, TextBlock(text="check.")],
    )
    assert message.text == "Let me check."
    assert message.tool_calls == [call]


def test_json_round_trip_restores_block_types() -> None:
    message = Message(
        role="assistant",
        content=[
            TextBlock(text="hi"),
            ToolCall(id="c1", name="search", arguments={"q": "x"}),
            ToolResult(call_id="c1", content="boom", is_error=True),
        ],
    )
    restored = Message.model_validate_json(message.model_dump_json())
    assert restored == message
    assert [type(b) for b in restored.content] == [TextBlock, ToolCall, ToolResult]


def test_messages_are_immutable() -> None:
    message = Message.user("hi")
    with pytest.raises(ValidationError):
        message.role = "assistant"


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ToolResult.model_validate({"call_id": "c1", "content": "ok", "is_eror": True})


def test_usage_adds() -> None:
    total = Usage(input_tokens=3, output_tokens=1) + Usage(input_tokens=2, output_tokens=4)
    assert total == Usage(input_tokens=5, output_tokens=5)
