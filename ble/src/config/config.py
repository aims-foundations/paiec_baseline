"""Settings used by the baseline's single prediction loop.

The predictor LLM is distinct from the subject whose correctness is predicted.
max_steps counts LLM turns, including the final submission turn.
"""

import math
from dataclasses import asdict, dataclass


@dataclass
class AgentConfig:
    llm: str = "openai/gpt-5.6-luna"
    max_tokens: int = 32000
    max_steps: int = 10
    question_timeout: float = 240
    reasoning_effort: str = "medium"

    def __post_init__(self):
        if not self.llm:
            raise ValueError("llm must name the predictor model")
        if self.max_steps < 1 or self.max_tokens < 1:
            raise ValueError("max_steps and max_tokens must be positive")
        if not math.isfinite(self.question_timeout) or self.question_timeout <= 0:
            raise ValueError("question_timeout must be positive and finite")
        if self.reasoning_effort not in {
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("unsupported predictor reasoning_effort")

    def to_dict(self) -> dict:
        return asdict(self)
