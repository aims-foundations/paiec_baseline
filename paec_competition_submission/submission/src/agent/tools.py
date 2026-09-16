"""Belief updates, final submission, and dispatch to the databank tools.

Tool argument schemas describe the interface the LLM must follow. Every action
contains updated_belief: the LLM's estimate based on evidence already seen.
Retrieval implementations live in comp_pipeline/script/agent_tools.py.
"""

import os

from comp_pipeline.script.databank import MeasurementDB

from agent.belief_state import (
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    BeliefState,
    clip_probability,
)

_BELIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "p": {
            "type": "number",
            "description": (
                f"Current probability ({MIN_PROBABILITY}-{MAX_PROBABILITY}). "
                "Required on every action, even when unchanged."
            ),
        },
        "base_rate_anchor": {
            "type": "string",
            "description": "Reference-class estimate and rationale",
        },
        "evidence_for": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Cumulative evidence for correctness; cite (tool_name, step_X).",
        },
        "evidence_against": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Cumulative evidence against correctness; cite (tool_name, step_X).",
        },
        "key_uncertainties": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Open questions that would most change the estimate.",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Your assessment of how well the evidence supports the estimate.",
        },
        "update_reasoning": {
            "type": "string",
            "description": "Explain how evidence already seen changed your probability.",
        },
    },
    "required": ["p", "confidence", "update_reasoning"],
}

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": "Submit your final probability and explanation; ends the loop.",
        "parameters": {
            "type": "object",
            "properties": {
                "probability": {
                    "type": "number",
                    "description": f"Final probability ({MIN_PROBABILITY}-{MAX_PROBABILITY})",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Summary of the evidence and inference",
                },
                "updated_belief": _BELIEF_SCHEMA,
            },
            "required": ["probability", "reasoning", "updated_belief"],
        },
    },
}


def get_tool_schemas() -> list:
    """Offer submit plus the retrieval tools supported by the local payload."""
    # Imported here because retrieval schemas reuse _BELIEF_SCHEMA above.
    from comp_pipeline.script.agent_tools import get_measurement_tools

    return [SUBMIT_TOOL, *get_measurement_tools()]


def parse_belief_update(arguments: dict, state: BeliefState) -> BeliefState:
    """Store the LLM's estimate; no probability is inferred by this function.

    Required fields must be present in a nested updated_belief object. Optional
    evidence fields retain their previous values when omitted. When provided,
    evidence lists must restate the accumulated evidence, as instructed in the
    prompt. Malformed updates raise rather than reusing the previous probability.
    """
    update = arguments["updated_belief"]
    if not isinstance(update, dict):
        raise TypeError("updated_belief must be a JSON object")
    confidence = update["confidence"]
    if confidence not in ("low", "medium", "high"):
        raise ValueError("confidence must be low, medium, or high")
    for field in ("base_rate_anchor", "update_reasoning"):
        if field in update and not isinstance(update[field], str):
            raise ValueError(f"updated_belief.{field} must be a string")
    for field in ("evidence_for", "evidence_against", "key_uncertainties"):
        if field in update:
            entries = update[field]
            if not isinstance(entries, list) or not all(
                isinstance(e, str) for e in entries
            ):
                raise ValueError(f"updated_belief.{field} must be a list of strings")
    return BeliefState(
        p=clip_probability(update["p"]),
        base_rate_anchor=update.get("base_rate_anchor", state.base_rate_anchor),
        evidence_for=update.get("evidence_for", state.evidence_for),
        evidence_against=update.get("evidence_against", state.evidence_against),
        key_uncertainties=update.get("key_uncertainties", state.key_uncertainties),
        confidence=confidence,
        update_reasoning=update["update_reasoning"],
        step=state.step,
    )


def dispatch_tool(
    name: str,
    arguments: dict,
    state: BeliefState,
    database: MeasurementDB,
    search_dir: str,
) -> tuple[str, BeliefState, dict]:
    """Store the pre-action belief, then submit or retrieve evidence.

    Returned retrieval text is read by the LLM on its next turn. A retrieval
    does not independently alter p. The submit action's probability is the
    final estimate and takes precedence over its accompanying updated_belief.p.
    Retrieval receives the shared public database directly, without question
    metadata or an intermediate configuration file.
    """
    from comp_pipeline.script.agent_tools import dispatch_measurement_tool

    state = parse_belief_update(arguments, state)
    if name == "submit":
        state.p = clip_probability(arguments["probability"])
        reasoning = arguments["reasoning"]
        if not isinstance(reasoning, str):
            raise ValueError("submit.reasoning must be a string")
        return (
            f"Submitted probability: {state.p:.4f}",
            state,
            {"reasoning": reasoning},
        )

    evidence = dispatch_measurement_tool(name, arguments, database)
    # Persist retrieval text for inspection; prediction never reads these logs.
    filename = f"tool_{name}_{state.step}.txt"
    os.makedirs(search_dir, exist_ok=True)
    with open(os.path.join(search_dir, filename), "w") as handle:
        handle.write(evidence)
    return evidence, state, {"result_file": filename}
