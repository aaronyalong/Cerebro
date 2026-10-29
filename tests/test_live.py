"""Tests against the real Claude API. Deselected by default; run with:

uv run --env-file .env pytest -m live
"""

import os

import pytest

from cerebro import Runner, ToolCall
from cerebro.providers.claude import ClaudeClient
from examples.notes_agent.agent import notes_agent

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"),
]


async def test_notes_agent_answers_from_its_tools() -> None:
    result = await Runner(ClaudeClient()).run(notes_agent, "When is the Alfred first milestone?")

    assert result.stop == "completed"
    tool_calls = [b for m in result.messages for b in m.content if isinstance(b, ToolCall)]
    assert tool_calls, "expected the agent to look the answer up"
    # The date only exists in the notes, so finding it proves the tools were used.
    assert "March" in result.final_text
    assert "14" in result.final_text
