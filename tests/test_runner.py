import pytest

from cerebro import Agent, Message, Runner, TextBlock, ToolCall, ToolResult, Usage, tool
from tests.fakes import FakeModelClient, reply, tool_use


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b


AGENT = Agent(
    name="calc",
    system_prompt="You are a calculator.",
    model="test-model",
    tools=[add, divide],
    max_tokens=100,
    max_iterations=3,
)


def call(id: str, name: str, **arguments: object) -> ToolCall:
    return ToolCall(id=id, name=name, arguments=arguments)


async def test_plain_answer_completes_in_one_step() -> None:
    client = FakeModelClient([reply("Hello!")])
    result = await Runner(client).run(AGENT, "Hi")

    assert result.stop == "completed"
    assert result.final_text == "Hello!"
    assert result.iterations == 1
    assert len(result.messages) == 2

    request = client.requests[0]
    assert request.model == "test-model"
    assert request.system == "You are a calculator."
    assert request.max_tokens == 100
    assert [t.name for t in request.tools] == ["add", "divide"]
    assert request.messages == [Message.user("Hi")]


async def test_tool_call_result_is_fed_back() -> None:
    client = FakeModelClient([tool_use(call("c1", "add", a=2, b=3)), reply("It's 5.")])
    result = await Runner(client).run(AGENT, "2+3?")

    assert result.stop == "completed"
    assert result.final_text == "It's 5."
    assert result.iterations == 2

    # Requests are snapshots: the first one only saw the user prompt.
    assert len(client.requests[0].messages) == 1
    tool_message = client.requests[1].messages[-1]
    assert tool_message.role == "user"
    assert tool_message.content == [ToolResult(call_id="c1", content="5")]


async def test_multiple_tool_calls_return_in_one_message_in_order() -> None:
    client = FakeModelClient(
        [tool_use(call("c1", "add", a=1, b=1), call("c2", "add", a=2, b=2)), reply("2 and 4")]
    )
    await Runner(client).run(AGENT, "Two sums")

    tool_message = client.requests[1].messages[-1]
    assert tool_message.content == [
        ToolResult(call_id="c1", content="2"),
        ToolResult(call_id="c2", content="4"),
    ]


async def test_tool_error_is_sent_back_so_the_model_can_recover() -> None:
    client = FakeModelClient(
        [
            tool_use(call("c1", "divide", a=1, b=0)),
            tool_use(call("c2", "divide", a=1, b=2)),
            reply("It's 0.5."),
        ]
    )
    result = await Runner(client).run(AGENT, "1/0, or else 1/2")

    error = client.requests[1].messages[-1].content[0]
    assert isinstance(error, ToolResult)
    assert error.is_error
    assert "ZeroDivisionError" in error.content
    assert result.stop == "completed"
    assert result.final_text == "It's 0.5."


async def test_stops_at_max_iterations() -> None:
    client = FakeModelClient([tool_use(call(f"c{i}", "add", a=1, b=1)) for i in range(3)])
    result = await Runner(client).run(AGENT, "Loop forever")

    assert result.stop == "max_iterations"
    assert result.iterations == 3
    assert len(client.requests) == 3
    # The last tool results are in the transcript even though the model never saw them.
    assert result.messages[-1].role == "user"


async def test_stops_when_reply_is_truncated() -> None:
    client = FakeModelClient([reply("Once upon a", stop_reason="max_tokens")])
    result = await Runner(client).run(AGENT, "Tell me a story")

    assert result.stop == "max_tokens"
    assert result.final_text == "Once upon a"


async def test_usage_is_summed_across_steps() -> None:
    client = FakeModelClient([tool_use(call("c1", "add", a=1, b=1)), reply("2")])
    result = await Runner(client).run(AGENT, "1+1")

    assert result.usage == Usage(input_tokens=20, output_tokens=10)


async def test_continues_an_existing_conversation_without_mutating_it() -> None:
    history = [
        Message.user("Hi"),
        Message(role="assistant", content=[TextBlock(text="Hello!")]),
        Message.user("What's 1+1?"),
    ]
    client = FakeModelClient([reply("2")])
    result = await Runner(client).run(AGENT, history)

    assert client.requests[0].messages == history
    assert len(history) == 3
    assert result.messages == [*history, result.messages[-1]]


def test_run_sync() -> None:
    result = Runner(FakeModelClient([reply("Hello!")])).run_sync(AGENT, "Hi")
    assert result.final_text == "Hello!"


def test_agent_rejects_zero_iterations() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        Agent(name="a", system_prompt="", model="m", max_iterations=0)
