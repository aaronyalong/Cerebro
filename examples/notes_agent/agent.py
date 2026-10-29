"""A notes assistant: answers questions using read-only tools over a folder of Markdown notes.

Run it (needs ANTHROPIC_API_KEY in .env):

    uv run --env-file .env python -m examples.notes_agent.agent "When is the Alfred milestone?"
"""

import sys
from pathlib import Path
from typing import Annotated

from pydantic import Field

from cerebro import Agent, Runner, RunResult, ToolCall, ToolResult, tool
from cerebro.providers.claude import ClaudeClient

NOTES_DIR = (Path(__file__).parent / "notes").resolve()
MODEL = "claude-haiku-5-5"

NoteName = Annotated[str, Field(description="Note name, as returned by list_notes")]
SearchQuery = Annotated[str, Field(description="Case-insensitive text to find")]


@tool
def list_notes() -> list[str]:
    """List the names of all notes."""
    return sorted(path.stem for path in NOTES_DIR.glob("*.md"))


@tool
def read_note(name: NoteName) -> str:
    """Read the full text of one note."""
    path = (NOTES_DIR / f"{name}.md").resolve()
    # Refuse anything that escapes the notes folder (e.g. "../../.env").
    if path.parent != NOTES_DIR or not path.is_file():
        raise ValueError(f"No note named {name!r}. Use list_notes to see what exists.")
    return path.read_text()


@tool
def search_notes(query: SearchQuery) -> list[str]:
    """Find lines containing the query across all notes, as "note: line" strings."""
    matches: list[str] = []
    for path in sorted(NOTES_DIR.glob("*.md")):
        for line in path.read_text().splitlines():
            if query.lower() in line.lower():
                matches.append(f"{path.stem}: {line.strip()}")
    return matches


notes_agent = Agent(
    name="notes",
    system_prompt=(
        "You answer questions using the user's notes. Look things up with the tools "
        "before answering, and say so if the notes don't contain the answer. Be concise."
    ),
    model=MODEL,
    tools=[list_notes, read_note, search_notes],
    max_iterations=8,
)


def print_transcript(result: RunResult) -> None:
    for message in result.messages[1:]:
        for block in message.content:
            if isinstance(block, ToolCall):
                print(f"-> {block.name}({block.arguments})")
            elif isinstance(block, ToolResult):
                marker = "!!" if block.is_error else "<-"
                print(f"{marker} {block.content[:100]}")
    print(f"\n{result.final_text}\n")
    print(f"[stop={result.stop} iterations={result.iterations} {result.usage}]")


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What is MCP, and why does it matter for agents?"
    result = Runner(ClaudeClient()).run_sync(notes_agent, question)
    print_transcript(result)


if __name__ == "__main__":
    main()
