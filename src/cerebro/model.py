"""The interface every model provider adapter implements."""

from typing import Literal, Protocol

from cerebro.types import FrozenModel, Message, ToolSpec, Usage

StopReason = Literal["end_turn", "tool_use", "max_tokens", "other"]


class ModelRequest(FrozenModel):
    model: str
    system: str
    messages: list[Message]
    tools: list[ToolSpec] = []
    max_tokens: int


class ModelResponse(FrozenModel):
    message: Message
    stop_reason: StopReason
    usage: Usage


class ModelClient(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse: ...
