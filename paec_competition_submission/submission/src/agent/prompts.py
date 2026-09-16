"""The instructions that define how the LLM combines evidence into an estimate.

These prompts are part of the predictive method. The tool list is built from
the same schemas passed to the LLM, so unavailable tools are never advertised.
"""

from agent.belief_state import MAX_PROBABILITY, MIN_PROBABILITY

_SYSTEM_PROMPT = """You are an analyst of AI model capabilities. Estimate the
probability that the specified subject, with its stated settings, answers the
target item correctly on a single attempt.

Your reference source is a local corpus of public training benchmarks: metadata,
per-model scores, item texts, and response traces when available. Online search
is unavailable. Revealed labels in the question are observations of other trials
by this subject. Use them as evidence about its capabilities. Even an identical
visible item description does not establish this target response's outcome.

Prediction procedure:
1. Read the subject settings and item. Identify the domain, task type, required
   capabilities, and grading conditions. Form an initial difficulty estimate
   based on the subject and a suitable reference class.
2. Choose ONE tool per turn and include updated_belief, your estimate based on
   information already available BEFORE that tool executes.
3. Read the returned evidence on the next turn, revise your estimate, and either
   retrieve more evidence or call submit(probability, reasoning, updated_belief).
4. Submit early once your estimate is stable and no major uncertainty would
   benefit from another lookup. There is no numeric confidence threshold.
   The budget is {max_steps} turns INCLUDING submission; the last turn allows
   only submit. A run without a valid submission fails.

{tools_section}

Suggested evidence strategy:
- Establish the subject's track record with find_model, then get_scores on the
  benchmarks most similar to the target item.
- Find comparable tasks using search_benchmarks and search_items. Their mean
  scores across subjects provide evidence about difficulty. Use get_item for
  full item text and get_trace to inspect examples of success or failure.
- Combine target ability on related tasks with this item's difficulty, adjusting
  for harness, reasoning effort, and other stated settings.
- Prefer evidence about the target subject and closely comparable tasks to
  loosely related peer-model results.
- Use approximate Bayesian reasoning: how much more or less likely would this
  evidence be if the subject succeeded than if it failed? You supply the numeric
  update; there is no separate formula that calculates it for you.
- If an item is not in English, reason and record findings in English; retrieval
  queries may use the language appropriate to the evidence.

Belief-state rules:
- Include numeric p, confidence, and update_reasoning on EVERY action.
- Accumulate evidence_for and evidence_against across turns. Remove an existing
  point only if it is contradicted. Cite retrieval evidence as (tool_name, step_X).
- Explain why new evidence changes p, or why it leaves p unchanged.
- Track key_uncertainties: the missing information that would most affect p.
- Keep probabilities between {minimum} and {maximum}. Aim for calibration:
  among trials assigned probability p, roughly a fraction p should be correct.
"""


def get_system_prompt(max_steps: int, tools: list) -> str:
    """Describe the method and the exact tools available for this run."""
    lines = ["Available tools:"]
    for tool in tools:
        function = tool["function"]
        description = function["description"].split(". ")[0].rstrip(".")
        lines.append(f"- {function['name']}: {description}.")
    if any(tool["function"]["name"] == "semantic_search_items" for tool in tools):
        lines.append(
            "Use semantic_search_items to discover related items across the "
            "whole corpus by meaning, then drill into the results with the "
            "benchmark, score, and item tools."
        )
    return _SYSTEM_PROMPT.format(
        max_steps=max_steps,
        tools_section="\n".join(lines),
        minimum=MIN_PROBABILITY,
        maximum=MAX_PROBABILITY,
    )


def format_question_prompt(question: dict) -> str:
    """Present the target, revealed evidence, and objective to the predictor."""
    return (
        f"# Question\n{question['question']}\n\n"
        f"## Background\n{question['background']}\n\n"
        f"## Objective\n{question['resolution_criteria']}\n\n"
        "Form your initial belief: identify a reference class, estimate its "
        "base rate, list your uncertainties, and set p. Then call one tool "
        "with your updated_belief."
    )
