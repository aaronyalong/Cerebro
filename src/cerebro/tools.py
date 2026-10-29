"""Turn typed Python functions into tools a model can call."""

import asyncio
import inspect
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, create_model

from cerebro.types import ToolCall, ToolResult, ToolSpec

# Both Anthropic and OpenAI require tool names to match this.
_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Serializes any return value (dicts, pydantic models, dates, ...) to JSON.
_ANY_ADAPTER: TypeAdapter[Any] = TypeAdapter(Any)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[..., Any]

    @classmethod
    def from_function(
        cls, fn: Callable[..., Any], name: str | None = None, description: str | None = None
    ) -> "Tool":
        name = name or fn.__name__
        if not _NAME_PATTERN.match(name):
            raise ValueError(f"Tool name {name!r} must match {_NAME_PATTERN.pattern}")
        description = description or _first_paragraph(inspect.getdoc(fn))
        if not description:
            raise ValueError(f"Tool {name!r} needs a docstring or an explicit description")
        return cls(name=name, description=description, input_model=_input_model(fn), fn=fn)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
        )


def tool(fn: Callable[..., Any]) -> Tool:
    """Decorator: `@tool` turns a typed, documented function into a Tool."""
    return Tool.from_function(fn)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            if t.name in self._tools:
                raise ValueError(f"Duplicate tool name {t.name!r}")
            self._tools[t.name] = t

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        """Run one tool call. Never raises: every failure becomes an error result."""
        t = self._tools.get(call.name)
        if t is None:
            available = ", ".join(self._tools) or "none"
            return _error(call, f"Unknown tool {call.name!r}. Available tools: {available}")

        try:
            args = t.input_model.model_validate(call.arguments)
        except ValidationError as e:
            return _error(call, f"Invalid arguments for {call.name!r}: {_format_errors(e)}")

        # Pass the validated values (not model_dump) so nested models stay models.
        kwargs = {field: getattr(args, field) for field in type(args).model_fields}
        try:
            if inspect.iscoroutinefunction(t.fn):
                result = await t.fn(**kwargs)
            else:
                result = await asyncio.to_thread(t.fn, **kwargs)
            content = _to_text(result)
        except Exception as e:
            return _error(call, f"{type(e).__name__}: {e}")
        return ToolResult(call_id=call.id, content=content)

    async def execute_all(self, calls: Iterable[ToolCall]) -> list[ToolResult]:
        """Run calls concurrently; results come back in the same order as the calls."""
        return list(await asyncio.gather(*(self.execute(c) for c in calls)))


def _input_model(fn: Callable[..., Any]) -> type[BaseModel]:
    """Build a pydantic model whose fields are the function's parameters."""
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(f"{fn.__name__}: *args/**kwargs are not supported in tools")
        if param.name not in hints:
            raise TypeError(f"{fn.__name__}: parameter {param.name!r} needs a type hint")
        default = ... if param.default is param.empty else param.default
        fields[param.name] = (hints[param.name], default)
    return create_model(
        f"{fn.__name__}_input",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _first_paragraph(doc: str | None) -> str:
    return doc.split("\n\n")[0].strip() if doc else ""


def _format_errors(error: ValidationError) -> str:
    parts: list[str] = []
    for err in error.errors(include_url=False):
        location = ".".join(str(p) for p in err["loc"]) or "(arguments)"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)


def _to_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    return _ANY_ADAPTER.dump_json(result).decode()


def _error(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(call_id=call.id, content=message, is_error=True)
