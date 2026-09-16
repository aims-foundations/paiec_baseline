"""Optional startup checks using fixed public input, before hidden evaluation."""

from .openai_api import OpenAIRequestError, post_json


def check_api_configuration(config):
    """Check both configured APIs once; results never affect predictions.

    Import-time errors can be reported without exposing hidden inputs. Only
    fixed provider codes, HTTP statuses, and exception types enter our summary.
    """
    from agent.llm_client import complete_turn
    from agent.tools import SUBMIT_TOOL

    from .embedding_index import get_index

    if config.llm != "openai/gpt-5.6-luna":
        raise ValueError("api_preflight currently supports the Luna predictor")
    try:
        response = complete_turn(
            [
                {
                    "role": "user",
                    "content": (
                        "This is a fixed public API connectivity test. A fair coin "
                        "is flipped once. Use submit to estimate the probability "
                        "of heads, with a short explanation and updated_belief."
                    ),
                }
            ],
            [SUBMIT_TOOL],
            config,
            remaining_seconds=60,
            force_submit=True,
        )
        calls = response.choices[0].message.tool_calls
        if len(calls) != 1 or calls[0].function.name != "submit":
            raise RuntimeError("OpenAI startup check returned no submit action")
        index = get_index()
        if index is None:
            raise FileNotFoundError("Startup check needs embeddings/manifest.json")
        manifest = index.manifest()
        result = post_json(
            "embeddings",
            {
                "model": manifest["model"],
                "input": ["A fixed public API connectivity test."],
                "dimensions": manifest["dimensions"],
                "encoding_format": "float",
            },
            timeout=60,
        )
        if len(result["data"][0]["embedding"]) != manifest["dimensions"]:
            raise ValueError("OpenAI startup check returned wrong embedding dimensions")
    except OpenAIRequestError as exc:
        raise RuntimeError(f"Public startup check failed: {exc.safe_summary}") from None
