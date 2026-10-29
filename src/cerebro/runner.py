"""The agent loop: call the model, run its tool calls, feed results back, repeat."""

import asyncio
from typing import Literal

from cerebro.agent import Agent
from cerebro.model import ModelClient, ModelRequest, ModelResponse
from cerebro.tools import ToolRegistry
from cerebro.types import FrozenModel, Message, Usage

StopKind = Literal["completed", "max_iterations", "max_tokens"]


class RunResult(FrozenModel):
    messages: list[Message]  # the full transcript, including the input
    stop: StopKind
    usage: Usage
    iterations: int

    @property
    def final_text(self) -> str:
        """Text of the last assistant message, or "" if there is none."""
        for message in reversed(self.messages):
            if message.role == "assistant":
                return message.text
        return ""


class Runner:
    def __init__(self, client: ModelClient) -> None:
        self.client = client

    async def run(self, agent: Agent, user_input: str | list[Message]) -> RunResult:
        """Run the agent until it answers without tool calls or hits a limit.

        `user_input` is either a new user prompt, or a full conversation to continue
        (for example `previous_result.messages + [Message.user("...")]`).
        """
        registry = ToolRegistry(agent.tools)
        messages = [Message.user(user_input)] if isinstance(user_input, str) else list(user_input)
        usage = Usage()
        stop: StopKind = "max_iterations"
        iterations = 0

        while iterations < agent.max_iterations:
            iterations += 1
            response = await self._call_model(agent, registry, messages)
            messages.append(response.message)
            usage += response.usage

            if response.stop_reason == "max_tokens":
                # The reply was cut off; any tool call in it may be incomplete.
                stop = "max_tokens"
                break

            calls = response.message.tool_calls
            if not calls:
                stop = "completed"
                break

            results = await registry.execute_all(calls)
            messages.append(Message(role="user", content=[*results]))

        return RunResult(messages=messages, stop=stop, usage=usage, iterations=iterations)

    def run_sync(self, agent: Agent, user_input: str | list[Message]) -> RunResult:
        """Blocking wrapper around `run` for scripts and quick tests.

        Don't call this from inside a running event loop (Jupyter, FastAPI);
        use `await runner.run(...)` there.
        """
        return asyncio.run(self.run(agent, user_input))

    async def _call_model(
        self, agent: Agent, registry: ToolRegistry, messages: list[Message]
    ) -> ModelResponse:
        request = ModelRequest(
            model=agent.model,
            system=agent.system_prompt,
            messages=messages,
            tools=registry.specs(),
            max_tokens=agent.max_tokens,
        )
        return await self.client.complete(request)
