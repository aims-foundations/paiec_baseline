"""One LLM request for one prediction turn.

The main reasoning loop calls this function. Semantic retrieval separately
uses the embedding API in comp_pipeline/script/embedding_index.py.
"""

import copy
import json
import time
from types import SimpleNamespace

from comp_pipeline.script.openai_api import post_json
from config.config import AgentConfig


def complete_turn(
    messages: list,
    tools: list,
    config: AgentConfig,
    remaining_seconds: float,
    force_submit: bool,
):
    """Ask for one tool call; reserve the final turn for submit.

    Persistent API failures propagate. The OpenAI transport retries temporary
    failures within this turn's timeout; the loop checks its deadline afterward.
    """
    if config.llm == "openai/gpt-5.6-luna":
        return _complete_responses_turn(
            messages, tools, config, remaining_seconds, force_submit
        )

    # Optional provider support is imported only when explicitly selected.
    # Luna's competition path requires no inference SDKs.
    import litellm

    kwargs = {
        "model": config.llm,
        "messages": messages,
        "tools": tools,
        "max_tokens": config.max_tokens,
        "timeout": min(120, remaining_seconds),
        "num_retries": 0,
    }
    if force_submit:
        kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": "submit"},
        }
    # The loop expects one action per turn. Other providers receive the same
    # instruction in the prompt; responses with multiple actions are rejected.
    if config.llm.startswith("openai/"):
        kwargs["parallel_tool_calls"] = False
    return litellm.completion(**kwargs)


def _complete_responses_turn(messages, tools, config, remaining_seconds, force_submit):
    """Use Luna's reasoning-capable endpoint and retain native output for replay.

    Each prediction owns its message history. Replaying the complete native
    output preserves encrypted reasoning, tool call IDs, and assistant text
    without requiring server-side response storage or a shared conversation.
    """
    deadline = time.monotonic() + remaining_seconds
    inputs = []
    for message in messages:
        native_output = (message.get("provider_specific_fields") or {}).get(
            "responses_output"
        )
        if native_output is not None:
            inputs.extend(native_output)
        elif message["role"] == "tool":
            inputs.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
        else:
            inputs.append({"role": message["role"], "content": message["content"]})

    payload = {
        "model": config.llm.removeprefix("openai/"),
        "input": inputs,
        "tools": [
            {
                "type": "function",
                **tool["function"],
                "parameters": _strict_schema(tool["function"]["parameters"]),
                "strict": True,
            }
            for tool in tools
        ],
        "reasoning": {"effort": config.reasoning_effort},
        "max_output_tokens": config.max_tokens,
        "parallel_tool_calls": False,
        "tool_choice": {"type": "function", "name": "submit"}
        if force_submit
        else "required",
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for attempt in range(1, 4):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("prediction deadline expired during Responses recovery")
        response = post_json("responses", payload, timeout=remaining)
        for key in usage_totals:
            usage_totals[key] += (response.get("usage") or {}).get(key, 0)
        if response.get("status") == "completed":
            break

        # Only an explicit token-limit result permits a larger generation.
        # Repeat the same turn without executing or replaying truncated calls.
        # Reasoning effort, prior completed history, and the deadline stay fixed.
        reason = (response.get("incomplete_details") or {}).get("reason")
        if (response.get("status") == "incomplete" and reason == "max_output_tokens"
                and attempt < 3 and payload["max_output_tokens"] < 128000):
            payload = {**payload, "max_output_tokens": min(128000, payload["max_output_tokens"] * 2)}
            continue
        raise RuntimeError(_incomplete_response_diagnostic(response, attempt))

    output = response["output"]
    schemas = {
        tool["function"]["name"]: tool["function"]["parameters"] for tool in tools
    }
    tool_calls = [
        SimpleNamespace(
            id=item["call_id"],
            function=SimpleNamespace(
                name=item["name"],
                arguments=_restore_optional_arguments(
                    item["arguments"], schemas.get(item["name"], {})
                ),
            ),
        )
        for item in output
        if item["type"] == "function_call"
    ]
    content = "\n".join(
        part["text"]
        for item in output
        if item["type"] == "message"
        for part in item["content"]
        if part["type"] == "output_text"
    )

    usage = usage_totals
    # Normalize only the interface consumed by the provider-independent loop.
    # Preserve every native output item separately for the next Responses call.
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                    provider_specific_fields={"responses_output": output},
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=usage["input_tokens"],
            completion_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
        ),
    )


def _incomplete_response_diagnostic(response, attempts):
    """Expose fixed status/reason names and counts, never generated content."""
    status = response.get("status")
    if status not in {"incomplete", "failed", "cancelled", "queued", "in_progress"}:
        status = "unknown"
    reason = (response.get("incomplete_details") or {}).get("reason")
    if reason not in {"max_output_tokens", "content_filter"}:
        reason = "unknown"
    usage = response.get("usage") or {}
    output_tokens = usage.get("output_tokens")
    if type(output_tokens) is not int or output_tokens < 0:
        output_tokens = "unknown"
    return (f"Responses API did not complete the turn: {status}; reason={reason}; "
            f"output_tokens={output_tokens}; attempts={attempts}")


def _strict_schema(schema):
    """Require every field on the wire; nullable optional fields mean omission."""
    schema = copy.deepcopy(schema)
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        for name, child in properties.items():
            strict_child = _strict_schema(child)
            if name not in required:
                strict_child = {"anyOf": [strict_child, {"type": "null"}]}
            properties[name] = strict_child
        schema["required"] = list(properties)
        schema["additionalProperties"] = False
    elif schema.get("type") == "array":
        schema["items"] = _strict_schema(schema["items"])
    return schema


def _restore_optional_arguments(arguments, schema):
    """Map explicit nulls in optional fields back to the tools' existing defaults.

    Required nulls and malformed values are preserved for the normal validation
    path to reject; only fields optional in the original schema may be omitted.
    The native function-call item is kept intact for reasoning-state replay.
    """

    def restore(value, definition):
        if not isinstance(value, dict) or definition.get("type") != "object":
            return value
        required = set(definition.get("required", []))
        properties = definition.get("properties", {})
        return {
            key: restore(child, properties.get(key, {}))
            for key, child in value.items()
            if child is not None or key in required or key not in properties
        }

    return json.dumps(restore(json.loads(arguments), schema))
