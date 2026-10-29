"""Claude adapter: translates between Cerebro's core types and the Anthropic Messages API."""

from typing import cast

from anthropic import AsyncAnthropic, omit
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaStopReason,
    BetaToolParam,
)
from anthropic.types.beta.beta_tool_param import InputSchema

from cerebro.model import ModelRequest, ModelResponse, StopReason
from cerebro.types import (
    ContentBlock,
    Message,
    ProviderBlock,
    TextBlock,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)

PROVIDER = "anthropic"

# Server-side refusal fallbacks: if the model declines for policy reasons, the API
# re-runs the request on a fallback model within the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeClient:
    def __init__(self, client: AsyncAnthropic | None = None, fallbacks: bool = True) -> None:
        # AsyncAnthropic() reads ANTHROPIC_API_KEY (or another configured credential).
        self._client = client or AsyncAnthropic()
        self._fallbacks = fallbacks

    async def complete(self, request: ModelRequest) -> ModelResponse:
        message = await self._client.beta.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            system=request.system or omit,
            messages=to_anthropic_messages(request.messages),
            tools=[to_anthropic_tool(spec) for spec in request.tools] or omit,
            betas=[FALLBACK_BETA] if self._fallbacks else omit,
            fallbacks="default" if self._fallbacks else omit,
        )
        return from_anthropic_message(message)


# --- core -> Anthropic ---


def to_anthropic_messages(messages: list[Message]) -> list[BetaMessageParam]:
    result: list[BetaMessageParam] = []
    for message in messages:
        blocks: list[BetaContentBlockParam] = []
        for block in message.content:
            param = _to_block_param(block)
            if param is not None:
                blocks.append(param)
        # Spelled out per role so the type checker can pick the matching SDK type.
        if message.role == "user":
            result.append({"role": "user", "content": blocks})
        else:
            result.append({"role": "assistant", "content": blocks})
    return result


def to_anthropic_tool(spec: ToolSpec) -> BetaToolParam:
    return {
        "name": spec.name,
        "description": spec.description,
        # Our schemas always come from a pydantic model, so they are JSON-schema objects.
        "input_schema": cast(InputSchema, spec.input_schema),
    }


def _to_block_param(block: ContentBlock) -> BetaContentBlockParam | None:
    match block:
        case TextBlock():
            return {"type": "text", "text": block.text}
        case ToolCall():
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.arguments,
            }
        case ToolResult():
            return {
                "type": "tool_result",
                "tool_use_id": block.call_id,
                "content": block.content,
                "is_error": block.is_error,
            }
        case ProviderBlock(provider=provider) if provider == PROVIDER:
            # Thinking, fallback markers, etc.: send back exactly what the API gave us.
            return cast(BetaContentBlockParam, block.data)
        case ProviderBlock():
            return None  # another provider's block means nothing to Claude


# --- Anthropic -> core ---


def from_anthropic_message(message: BetaMessage) -> ModelResponse:
    content: list[ContentBlock] = []
    for block in message.content:
        if block.type == "text":
            content.append(TextBlock(text=block.text))
        elif block.type == "tool_use":
            content.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        else:
            content.append(
                ProviderBlock(
                    provider=PROVIDER,
                    kind=block.type,
                    text=block.thinking if block.type == "thinking" else "",
                    # by_alias: wire names (e.g. "from", not "from_"), so it round-trips.
                    data=block.model_dump(mode="json", by_alias=True, exclude_none=True),
                )
            )
    return ModelResponse(
        message=Message(role="assistant", content=content),
        stop_reason=_stop_reason(message.stop_reason),
        usage=Usage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        ),
    )


def _stop_reason(reason: BetaStopReason | None) -> StopReason:
    match reason:
        case "end_turn" | "stop_sequence":
            return "end_turn"
        case "tool_use":
            return "tool_use"
        case "max_tokens" | "model_context_window_exceeded":
            return "max_tokens"
        case "refusal":
            return "refusal"
        case _:
            return "other"
