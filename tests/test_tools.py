import asyncio
from typing import Annotated

import pytest
from pydantic import Field

from cerebro import Tool, ToolCall, ToolRegistry, tool


@tool
def add(a: int, b: Annotated[int, Field(description="Second number")] = 1) -> int:
    """Add two numbers.

    This second paragraph is not part of the description.
    """
    return a + b


@tool
async def shout(text: str) -> str:
    """Upper-case some text."""
    return text.upper()


@tool
def fail() -> str:
    """Always fails."""
    raise ValueError("disk on fire")


def call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=arguments)


# --- schema generation ---


def test_spec_from_signature_and_docstring() -> None:
    spec = add.spec
    assert spec.name == "add"
    assert spec.description == "Add two numbers."
    schema = spec.input_schema
    assert schema["required"] == ["a"]
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["properties"]["b"]["description"] == "Second number"
    assert schema["properties"]["b"]["default"] == 1
    assert schema["additionalProperties"] is False


def test_missing_docstring_is_rejected() -> None:
    def undocumented(x: int) -> int:
        return x

    with pytest.raises(ValueError, match="docstring"):
        Tool.from_function(undocumented)


def test_missing_type_hint_is_rejected() -> None:
    def untyped(x) -> int:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
        """Untyped."""
        return 0

    with pytest.raises(TypeError, match="type hint"):
        Tool.from_function(untyped)  # pyright: ignore[reportUnknownArgumentType]


def test_var_args_are_rejected() -> None:
    def variadic(*items: int) -> int:
        """Variadic."""
        return sum(items)

    with pytest.raises(TypeError, match=r"\*args"):
        Tool.from_function(variadic)


def test_invalid_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="must match"):
        Tool.from_function(add.fn, name="has spaces")


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        ToolRegistry([add, add])


# --- execution ---


async def test_executes_sync_tool() -> None:
    result = await ToolRegistry([add]).execute(call("add", a=2, b=3))
    assert result.content == "5"
    assert not result.is_error
    assert result.call_id == "c1"


async def test_executes_async_tool() -> None:
    result = await ToolRegistry([shout]).execute(call("shout", text="hi"))
    assert result.content == "HI"


async def test_non_string_results_become_json() -> None:
    @tool
    def info() -> dict[str, int]:
        """Return a dict."""
        return {"n": 1}

    result = await ToolRegistry([info]).execute(call("info"))
    assert result.content == '{"n":1}'


async def test_unknown_tool_is_an_error_result() -> None:
    result = await ToolRegistry([add]).execute(call("nope"))
    assert result.is_error
    assert "Unknown tool 'nope'" in result.content
    assert "add" in result.content


async def test_invalid_arguments_are_an_error_result() -> None:
    result = await ToolRegistry([add]).execute(call("add", a="not a number"))
    assert result.is_error
    assert "a: Input should be a valid integer" in result.content


async def test_extra_arguments_are_an_error_result() -> None:
    result = await ToolRegistry([add]).execute(call("add", a=1, c=2))
    assert result.is_error
    assert "c: Extra inputs are not permitted" in result.content


async def test_tool_exception_is_an_error_result() -> None:
    result = await ToolRegistry([fail]).execute(call("fail"))
    assert result.is_error
    assert result.content == "ValueError: disk on fire"


async def test_execute_all_runs_concurrently_and_keeps_order() -> None:
    second_started = asyncio.Event()

    @tool
    async def first() -> str:
        """Waits for second; would time out if calls ran one after another."""
        await asyncio.wait_for(second_started.wait(), timeout=1)
        return "first"

    @tool
    async def second() -> str:
        """Unblocks first."""
        second_started.set()
        return "second"

    registry = ToolRegistry([first, second])
    calls = [
        ToolCall(id="1", name="first", arguments={}),
        ToolCall(id="2", name="second", arguments={}),
    ]
    results = await registry.execute_all(calls)
    assert [r.content for r in results] == ["first", "second"]
    assert not any(r.is_error for r in results)
