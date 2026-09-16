"""Bayesian Linguistic Evaluator: predict a subject's probability of success.

Builds a question from the supplied subject and item, then runs the
retrieval-based predictor. Revealed outcomes from the same subject provide
supporting evidence, never a direct answer shortcut.
"""

import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# Wire the vendored agent core and this pipeline's tool package.
for _p in (_HERE, os.path.join(_HERE, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load once at import: defaults, then local JSON, then environment overrides.
# Initialize credentials before importing libraries that make API requests.
CFG = {
    "llm": "openai/gpt-5.6-luna",
    "max_steps": 10,
    "question_timeout": 240,
    "reasoning_effort": "medium",
    "hf_data_repo": "",
    "api_keys": {},
}

try:
    with open(os.path.join(_HERE, "submission_config.json")) as _config_file:
        CFG.update(json.load(_config_file))
except FileNotFoundError:
    pass  # The local config is optional; malformed JSON still raises.

if os.environ.get("BLF_LLM"):
    CFG["llm"] = os.environ["BLF_LLM"]
if os.environ.get("BLF_MAX_STEPS"):
    CFG["max_steps"] = int(os.environ["BLF_MAX_STEPS"])

# Existing environment credentials take precedence over keys in the file.
for _key, _value in (CFG.get("api_keys") or {}).items():
    if _value and not os.environ.get(_key):
        os.environ[_key] = _value

from agent.agent import run_agent
from agent.belief_state import MAX_PROBABILITY, MIN_PROBABILITY
from config.config import AgentConfig

_LABELED_EXCERPT_CHARS = 300  # bound the context contributed by each revealed trial
_RESOLUTION_CRITERIA = (
    "Your probability estimate should be P(the target model answers this item "
    "correctly on a single attempt)."
)


def predict(input: list[dict], labeled: list | None = None) -> float:
    """Return P(correct), clipped to the baseline's [0.02, 0.98] bounds.

    `input` is `[subject, item]` (two dicts of strings); `labeled` is a list
    of `[[subject, item], label]` entries, possibly empty. Invalid inputs and
    prediction failures raise so the evaluator can report them clearly.
    """
    subject, item = input
    question = build_question(subject, item, labeled=labeled or [])
    # Configure the predictor LLM independently of the input subject.
    config = AgentConfig(
        llm=CFG["llm"],
        max_steps=int(CFG["max_steps"]),
        question_timeout=float(CFG["question_timeout"]),
        reasoning_effort=CFG["reasoning_effort"],
    )
    result = run_agent(question, config)
    if not result["submitted"]:
        raise RuntimeError("agent did not submit a prediction")

    probability = float(result["forecast"])
    if not math.isfinite(probability):
        raise ValueError(f"non-finite probability: {probability}")
    return min(max(probability, MIN_PROBABILITY), MAX_PROBABILITY)


def build_question(subject: dict, item: dict, labeled: list | None = None) -> dict:
    """Combine all target fields with compact examples from the same subject.

    Returns only the text the predictor needs. Corpus loading and run-log
    management belong to the retrieval layer and agent, respectively.

    Revealed examples must match the complete visible subject dictionary.
    Each includes only a verdict and item-text excerpt to keep context short;
    unlike the target item, examples omit item features and interactors.
    Excerpts provide evidence for reasoning, not keys for matching outcomes.
    """
    target = subject.get("normalized_name") or "unknown model"
    content = item.get("item_content", "")

    bg_lines = [
        "AI model's settings:",
        f"- Provider: {_format_field(subject.get('provider'))}",
        f"- Release date: {_format_field(subject.get('release_date'))}",
        f"- Access date: {_format_field(subject.get('access_date'))}",
        f"- Harness setting: {_format_field(subject.get('harness'))}",
        f"- Harness version: {_format_field(subject.get('harness_version'))}",
        f"- Reasoning effort level: {_format_field(subject.get('reasoning_effort'))}",
        f"- Other features (if available): {_format_field(subject.get('subject_features_extra'))}",
        "",
        "Item the subject must answer:",
        f"- Question: {_format_field(content)}",
        f"- Any features of the question (if available): {_format_field(item.get('item_features'))}",
        "",
        "Extra information (if available)",
        f"- Interactor (other components): {_format_field(item.get('interactors'))}",
    ]
    observations = []
    for (observed_subject, observed_item), label in labeled or []:
        if observed_subject != subject:
            continue
        verdict = "CORRECT" if float(label) >= 0.5 else "INCORRECT"
        excerpt = (
            " ".join(observed_item["item_content"].split())[:_LABELED_EXCERPT_CHARS]
            or "(item text unavailable)"
        )
        observations.append(f"- [{verdict}] {excerpt}")
    if observations:
        bg_lines += [
            "",
            (
                "Revealed outcomes from this subject on trials "
                "from the same hidden evaluation:"
            ),
        ] + observations

    return {
        "question": (
            f"What is the probability the AI model '{target}', "
            f"with the following settings, answers the item below "
            f"correctly (on a single attempt)?"
        ),
        "background": "\n".join(bg_lines),
        "resolution_criteria": _RESOLUTION_CRITERIA,
    }


def _format_field(value) -> str:
    """Render a metadata value, marking empty or missing fields explicitly."""
    text = str(value).strip() if value is not None else ""
    return text if text and text.lower() not in ("nan", "none") else "No information"


if CFG.get("api_preflight", False):
    from comp_pipeline.script.api_preflight import check_api_configuration

    check_api_configuration(
        AgentConfig(
            llm=CFG["llm"],
            max_steps=int(CFG["max_steps"]),
            question_timeout=float(CFG["question_timeout"]),
            reasoning_effort=CFG["reasoning_effort"],
        )
    )
