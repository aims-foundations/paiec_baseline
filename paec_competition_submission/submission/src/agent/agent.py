"""The baseline's complete reasoning loop: estimate, retrieve, revise, submit.

On each turn the LLM emits its current belief and chooses one action. A retrieval
returns evidence that the LLM can incorporate on its NEXT turn. Python records
the belief; it does not calculate a Bayesian update or decide when confidence is
high enough. The LLM can submit early, and the final allowed turn requires it.
"""

import json
import os
import tempfile
import time

from comp_pipeline.script.databank import get_db
from config.config import AgentConfig

from agent.belief_state import BeliefState
from agent.llm_client import complete_turn
from agent.prompts import format_question_prompt, get_system_prompt
from agent.tools import SUBMIT_TOOL, dispatch_tool, get_tool_schemas


def run_agent(question: dict, config: AgentConfig) -> dict:
    """Predict one response and return its forecast and inspectable reasoning trace.

    max_steps includes submission: a ten-turn run permits at most nine
    retrievals. belief_history starts with the initial placeholder and then
    records each LLM update; tool_log records the actions and retrieved text.

    The retrieval layer supplies the public database directly. The agent gives
    each invocation a fresh log directory under BLF_OUT/runs/searches, including
    repeated inputs. Run identifiers never enter the LLM's question or prompts.

    A result is returned only after a valid submit action. API/tool failures,
    malformed calls, and deadline expiry raise instead of returning a partial
    belief. The deadline is checked around operations, and LLM requests receive
    the remaining time as their timeout; it cannot interrupt local file reads.
    """
    started = time.monotonic()
    deadline = started + config.question_timeout
    state = BeliefState()
    belief_history = [state.to_dict()]
    tool_log = []
    tokens_in = tokens_out = 0
    database = get_db()
    output_root = os.environ.get(
        "BLF_OUT", os.path.join(tempfile.gettempdir(), "paec_blf_runs")
    )
    search_root = os.path.join(output_root, "runs", "searches")
    os.makedirs(search_root, exist_ok=True)
    search_dir = tempfile.mkdtemp(prefix="run_", dir=search_root)

    available_tools = get_tool_schemas()
    system_prompt = get_system_prompt(config.max_steps, available_tools)
    question_prompt = format_question_prompt(question)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_prompt},
    ]

    for step in range(1, config.max_steps + 1):
        state.step = step
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError("prediction deadline expired before submission")

        force_submit = step == config.max_steps
        tools = [SUBMIT_TOOL] if force_submit else available_tools
        response = complete_turn(
            messages,
            tools,
            config,
            remaining_seconds,
            force_submit,
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("prediction deadline expired during the LLM call")

        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if len(tool_calls) != 1:
            raise ValueError("each LLM turn must contain exactly one tool call")
        call = tool_calls[0]
        name = call.function.name
        if name not in {tool["function"]["name"] for tool in tools}:
            raise ValueError(f"tool {name!r} is not available on turn {step}")
        arguments = json.loads(call.function.arguments)
        if not isinstance(arguments, dict):
            raise TypeError("tool arguments must be a JSON object")

        # updated_belief describes knowledge BEFORE this action. The returned
        # evidence is appended to messages below for the next LLM turn.
        evidence, state, metadata = dispatch_tool(
            name,
            arguments,
            state,
            database,
            search_dir,
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("prediction deadline expired during a tool call")
        belief_history.append(state.to_dict())
        tokens_in += response.usage.prompt_tokens
        tokens_out += response.usage.completion_tokens
        tool_log.append(
            {
                "step": step,
                "tool": name,
                "args": {k: v for k, v in arguments.items() if k != "updated_belief"},
                "belief_p": state.p,
                "result": evidence,
                **metadata,
            }
        )

        if name == "submit":
            return {
                "run_id": os.path.basename(search_dir),
                "log_dir": search_dir,
                "forecast": state.p,
                "reasoning": metadata["reasoning"],
                "submitted": True,
                "system_prompt": system_prompt,
                "question_prompt": question_prompt,
                "belief_history": belief_history,
                "tool_log": tool_log,
                "n_steps": step,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "config": config.to_dict(),
            }

        assistant_message = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": call.function.arguments,
                    },
                }
            ],
        }
        # Keep the native output, including encrypted reasoning, alongside
        # tool results for the next Responses call within this prediction.
        native_output = (getattr(message, "provider_specific_fields", None) or {}).get(
            "responses_output"
        )
        if native_output is not None:
            assistant_message["provider_specific_fields"] = {
                "responses_output": native_output
            }

        messages.extend(
            [
                assistant_message,
                {"role": "tool", "tool_call_id": call.id, "content": evidence},
                {
                    "role": "user",
                    "content": (
                        f"{state.to_prompt_str(config.max_steps)}\n"
                        "Consider the new tool result when updating your next belief."
                    ),
                },
            ]
        )

    raise RuntimeError("agent exhausted its turns without submitting")
