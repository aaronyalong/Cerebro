"""A scripted ModelClient so loop tests never touch the network."""

from collections.abc import Iterable

from cerebro import Message, ModelRequest, ModelResponse, StopReason, TextBlock, ToolCall, Usage

STEP_USAGE = Usage(input_tokens=10, output_tokens=5)


class FakeModelClient:
    """Returns scripted responses in order and records every request it receives."""

    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("FakeModelClient ran out of scripted responses")
        return self._responses.pop(0)


def reply(text: str, stop_reason: StopReason = "end_turn") -> ModelResponse:
    return ModelResponse(
        message=Message(role="assistant", content=[TextBlock(text=text)]),
        stop_reason=stop_reason,
        usage=STEP_USAGE,
    )


def tool_use(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(
        message=Message(role="assistant", content=[*calls]),
        stop_reason="tool_use",
        usage=STEP_USAGE,
    )
