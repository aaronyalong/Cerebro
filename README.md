# Cerebro

A minimal, readable, model-agnostic agent harness in Python, built from scratch.

Cerebro turns a model plus a set of tools into a reliable agent. It owns the agent
loop, tool execution, context management, safety policies, tracing and evaluation,
and talks to model providers directly through their official SDKs — no agent
frameworks.

> **Status:** v0.1: Runner, Claude adapter and ToolRegistry working end to end.

## Why

Agent frameworks hide the parts that matter most when an agent misbehaves: what was
sent to the model, how tool calls were dispatched, why the loop stopped. Cerebro is a
learning and portfolio project whose goal is to understand and own those internals.
Every component is small enough to read in one sitting.

It also serves as the foundation for two downstream projects:

- **Alfred** — turns course files into practice tests.
- **Oracle** — local RAG over an Obsidian vault.

## Design principles

- **Model-agnostic core.** The harness uses its own message and tool-call types.
  Each provider adapter translates to and from them; nothing outside an adapter
  imports a provider SDK.
- **Hand-built core.** No LangChain, LangGraph, LlamaIndex, LiteLLM or similar.
  Dependencies are limited to official provider SDKs, pydantic and httpx.
- **Async throughout.** Tool calls within a turn run concurrently; `run_sync()` keeps
  scripts simple.
- **Clarity over cleverness.** Small modules, explicit data flow, fully typed (strict pyright).
- **Errors are data.** Bad tool arguments or failing tools are returned to the model
  as tool results so it can self-correct, instead of crashing the run.

## Architecture

| Component        | Responsibility                                                                 |
|------------------|--------------------------------------------------------------------------------|
| `Agent`          | Declarative config: system prompt, tools, model, policies.                     |
| `Runner`         | The loop: call model → dispatch tool calls → feed results back → stop. The only component that knows about all the others. |
| `ModelClient`    | Provider interface plus adapters (Claude first, then OpenAI-compatible, which covers OpenAI and Ollama). |
| `ToolRegistry`   | Generates JSON schemas from typed Python functions, validates arguments, reports errors back to the model. Later: MCP tools. |
| `ContextManager` | Keeps the conversation within a token budget (trimming, then compaction).      |
| `Policies`       | Pre/post-step hooks: iteration, token and time limits; approval for side-effecting tools. |
| `Tracer`         | One structured event per step, written as JSONL; optional Langfuse exporter.   |
| `Evals`          | Replay of saved traces plus a small task benchmark.                            |

Out of scope for now: graph/workflow DSL, UI, hosting.

## Getting started

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env   # then add your ANTHROPIC_API_KEY
uv sync
uv run pytest
```

## Usage

```python
from cerebro import Agent, Runner, tool
from cerebro.providers.claude import ClaudeClient


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


agent = Agent(
    name="calc",
    system_prompt="Use the tools to do arithmetic.",
    model="claude-haiku-5-5",
    tools=[add],
)

result = Runner(ClaudeClient()).run_sync(agent, "What is 19 + 23?")
print(result.final_text)   # also: result.stop, result.usage, result.messages
```

The tool's schema comes from its type hints and its description from the docstring.
Inside an event loop (FastAPI, Jupyter), use `await runner.run(...)` instead of `run_sync`.

A fuller example, a notes assistant with read-only file tools, is in
[examples/notes_agent](examples/notes_agent/agent.py):

```bash
uv run --env-file .env python -m examples.notes_agent.agent "When is the Alfred milestone?"
```

## Development

```bash
uv run pytest                            # unit tests (offline)
uv run --env-file .env pytest -m live    # tests against the real API
uv run ruff check . && uv run pyright    # lint + strict type check
```

## License

TBD.
