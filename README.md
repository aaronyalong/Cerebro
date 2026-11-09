# Cerebro

A minimal, readable, model-agnostic agent harness for Python.

Cerebro turns a model plus a set of typed Python functions into a working agent. It
owns the agent loop, tool execution and error handling, and talks to model providers
directly through their official SDKs. No agent framework sits underneath.

> **Status:** early development (v0.1). The core loop, tool registry and Claude adapter
> work end to end; the API may still change between releases.

## Why Cerebro

When an agent misbehaves, the questions are always the same: what was sent to the
model, how were the tool calls dispatched, and why did the loop stop? Large frameworks
make those answers hard to find. Cerebro keeps every component small enough to read in
one sitting, so you can see, and change, exactly what your agent does.

- **Model-agnostic core.** Cerebro has its own message and tool-call types. Each
  provider adapter translates to and from them, and nothing outside an adapter imports
  a provider SDK.
- **Minimal dependencies.** Just pydantic and the official provider SDKs.
- **Typed tools, no boilerplate.** A tool's JSON schema is generated from its type hints
  and its description from its docstring.
- **Errors are data.** Invalid arguments and failing tools are returned to the model as
  tool results, so it can correct itself instead of crashing the run.
- **Async throughout.** Tool calls within a turn run concurrently; `run_sync()` keeps
  scripts simple.
- **Fully typed.** Checked with pyright in strict mode.

## Installation

Requires Python 3.12+. Cerebro isn't on PyPI yet; install it from GitHub:

```bash
uv add git+https://github.com/aaronyalong/Cerebro
# or: pip install git+https://github.com/aaronyalong/Cerebro
```

The Claude adapter reads your key from the `ANTHROPIC_API_KEY` environment variable.

## Quickstart

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

Inside an event loop (FastAPI, Jupyter), use `await runner.run(...)` instead of
`run_sync`. To continue a conversation, pass `result.messages` plus a new user message
back to `run`.

## Examples

[`examples/notes_agent`](examples/notes_agent/agent.py) is a notes assistant that
answers questions using read-only tools over a folder of Markdown files:

```bash
uv run --env-file .env python -m examples.notes_agent.agent "What do my notes say about MCP?"
```

## Architecture

| Component        | Responsibility                                                              |
|------------------|-----------------------------------------------------------------------------|
| `Agent`          | Declarative config: system prompt, model, tools and limits.                 |
| `Runner`         | The loop: call the model, run its tool calls, feed results back, stop.      |
| `ToolRegistry`   | Schemas from typed functions, argument validation, concurrent execution.    |
| `ModelClient`    | Provider interface plus adapters: Claude, and OpenAI-compatible (OpenAI and Ollama). |
| `Tracer`         | One structured event per step, written as JSONL.                            |
| `Policies`       | Iteration, token and time limits; approval for side-effecting tools.        |
| `ContextManager` | Keeps long conversations within a token budget.                             |
| `Evals`          | Replay of saved traces plus a small task benchmark.                         |
| MCP              | Use tools from MCP servers.                                                 |

## Development

```bash
git clone https://github.com/aaronyalong/Cerebro && cd Cerebro
cp .env.example .env                     # add your ANTHROPIC_API_KEY (only for live tests)
uv sync
uv run pytest                            # unit tests (offline)
uv run --env-file .env pytest -m live    # tests against the real API
uv run ruff check . && uv run pyright    # lint + strict type check
```

## License

TBD.
