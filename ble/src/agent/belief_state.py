"""The LLM's running estimate and evidence, restated on each agent turn.

The LLM supplies the probability; Python stores and validates it. The initial
0.5 is a placeholder until the first LLM update, never a substitute prediction.
"""

import math
from dataclasses import asdict, dataclass, field

MIN_PROBABILITY = 0.02
MAX_PROBABILITY = 0.98


def clip_probability(value: float) -> float:
    """Reject non-finite values before clipping to the baseline's chosen bounds.

    Keeping estimates away from certainty is a baseline design choice, not a
    competition requirement. Belief updates and submission share this rule.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("probability must be a number")
    probability = float(value)
    if not math.isfinite(probability):
        raise ValueError(f"non-finite probability: {probability}")
    return min(max(probability, MIN_PROBABILITY), MAX_PROBABILITY)


@dataclass
class BeliefState:
    """Evidence lists are the LLM's cumulative account of previous retrievals."""

    p: float = 0.5
    base_rate_anchor: str = ""  # reference-class estimate and rationale
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    key_uncertainties: list[str] = field(default_factory=list)
    confidence: str = "low"  # self-assessment; no automatic stopping threshold
    update_reasoning: str = ""
    step: int = 0

    def to_prompt_str(self, max_steps: int) -> str:
        """Show the current estimate alongside the next turn's retrieved evidence."""
        lines = [
            f"Current belief state (turn {self.step}/{max_steps}):",
            f"  Probability: {self.p:.3f}",
            f"  Confidence: {self.confidence}",
        ]
        if self.base_rate_anchor:
            lines.append(f"  Reference-class estimate: {self.base_rate_anchor}")
        for title, entries in (
            ("Evidence FOR correctness", self.evidence_for),
            ("Evidence AGAINST correctness", self.evidence_against),
            ("Open uncertainties", self.key_uncertainties),
        ):
            if entries:
                lines.append(f"  {title}:")
                lines.extend(f"    - {entry}" for entry in entries)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)
