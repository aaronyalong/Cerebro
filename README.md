# Cerebro

A minimal, readable, model-agnostic agent harness in Python, built from scratch.

Cerebro turns a model plus a set of tools into a reliable agent. It owns the agent
loop, tool execution, context management, safety policies, tracing and evaluation,
and talks to model providers directly through their official SDKs — no agent
frameworks.

> **Status:** early development (working towards v0.1).

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
- **Clarity over cleverness.** Small modules, type hints throughout, explicit data flow.
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

Usage examples will be added as v0.1 lands.

## License

TBD.
