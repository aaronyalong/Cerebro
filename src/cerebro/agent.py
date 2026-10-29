"""Declarative agent configuration."""

from collections.abc import Sequence
from dataclasses import dataclass

from cerebro.tools import Tool


@dataclass(frozen=True)
class Agent:
    name: str
    system_prompt: str
    model: str
    tools: Sequence[Tool] = ()
    max_tokens: int = 16000
    # Hard cap on model calls per run. Moves into a policy in v0.2.
    max_iterations: int = 10

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
