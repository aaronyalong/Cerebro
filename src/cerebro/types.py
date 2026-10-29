"""Provider-agnostic conversation types.

Every provider adapter translates to and from these. Nothing here knows about any SDK.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    """Base for core types: immutable once created, and unknown fields are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class TextBlock(FrozenModel):
    type: Literal["text"] = "text"
    text: str


class ToolCall(FrozenModel):
    """The model asking to run a tool."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(FrozenModel):
    """The outcome of a tool call, sent back to the model."""

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[TextBlock | ToolCall | ToolResult, Field(discriminator="type")]

Role = Literal["user", "assistant"]


class Message(FrozenModel):
    role: Role
    content: list[ContentBlock]

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role="user", content=[TextBlock(text=text)])

    @property
    def text(self) -> str:
        """All text blocks joined together."""
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [block for block in self.content if isinstance(block, ToolCall)]


class ToolSpec(FrozenModel):
    """What the model is told about a tool."""

    name: str
    description: str
    input_schema: dict[str, Any]


class Usage(FrozenModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )
