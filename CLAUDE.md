# CLAUDE.md

Guidance for Claude Code sessions working in this repo. See README.md for the
project overview and roadmap.

## What this is

Cerebro is a from-scratch, model-agnostic agent harness in Python. It is a learning
and portfolio project: the point is to understand and own agent internals, so the
core is built by hand. Alfred and Oracle (the owner's other projects) will depend on it.

## Hard constraints

- **No agent frameworks.** No LangChain, LangGraph, LlamaIndex, LiteLLM or similar.
- **Allowed dependencies:** official provider SDKs (`anthropic`, `openai`), `pydantic`,
  `httpx`, plus dev tools (`pytest`, `ruff`). **Ask before adding any other dependency**,
  including dev dependencies.
- **Python 3.12+**, packaged with **uv**.
- **Model-agnostic core.** The core uses Cerebro's own message and tool-call types.
  Only provider adapters import provider SDKs; they translate to and from core types.
- **Secrets:** never commit them. API keys come from `.env` (gitignored).
  `.env.example` lists the expected variables without values.

## Working rules

- **Never run `git commit` or `git push`.** The owner commits. After each step:
  stage the relevant files with `git add`, show what is staged, and give a one-line
  commit message (subject only, imperative mood, no body, no trailers).
- Work in small steps. **Before each change, explain what you are about to do and
  why.** The owner wants to understand every design decision, not just receive code.
- Stay within the current milestone. Design interfaces so later milestones can plug
  in, but don't build them early.
- If something in a request seems wrong or underspecified, say so instead of guessing.

## Code conventions

- Type hints on everything, checked by **pyright in strict mode**. Keep code small and
  readable. Clarity over cleverness.
- **Async core.** `ModelClient`, tool execution and `Runner.run` are `async`; tool calls
  within one turn run concurrently. `Runner.run_sync()` is the convenience entry point for
  scripts. Tools may be sync (run via `asyncio.to_thread`) or async.
- The library never reads `.env`; that is the calling app's job. Examples and live tests
  use `uv run --env-file .env ...`.
- Core data types are pydantic models (or frozen dataclasses where validation isn't needed).
- Tool errors and invalid arguments are returned to the model as error tool results,
  not raised out of the loop.
- Unit tests never hit the network: use a scripted fake `ModelClient`. Tests that call
  a real provider are marked `live` and skipped unless explicitly requested and an API
  key is present.
- Lint and format with ruff.

## Commands

```bash
uv sync                 # install deps
uv run pytest           # unit tests (offline)
uv run --env-file .env pytest -m live   # live provider tests
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type check (strict)
```

## Layout

To be filled in once the v0.1 layout is approved.
