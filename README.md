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

## Roadmap

- [ ] **v0.1** — Runner + Claude adapter + ToolRegistry + one example agent, end to end, with tests
- [ ] **v0.2** — Tracer + Policies
- [ ] **v0.3** — ContextManager
- [ ] **v0.4** — Evals (replay + benchmark)
- [ ] **v0.5** — OpenAI-compatible adapter (Ollama) + MCP client; Alfred runs on Cerebro
- [ ] **v0.6** — Oracle runs on Cerebro; refactor whatever didn't generalize

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
