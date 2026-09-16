"""Check the baseline and optionally show a prediction's evidence trail.

    python tools/smoke_test.py --mock --walkthrough  # synthetic data, no API calls
    python tools/smoke_test.py --walkthrough         # prepared public data + real LLM

Mock mode uses in-memory tables and scripted probabilities. It exercises the
real prompt assembly, retrieval, belief updates, submission, and error handling.
Real mode makes one prediction with example revealed labels using the configured
provider and credentials, and checks acquisition decisions without extra API calls.
"""

import argparse
import copy
import io
import json
import os
import socket
import ssl
import sys
import tempfile
import threading
import types
from contextlib import ExitStack, contextmanager
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener

_HERE = str(Path(__file__).resolve().parent)
_SUB = os.path.normpath(os.path.join(_HERE, "..", "ble"))
_PAYLOAD = os.path.normpath(os.path.join(_HERE, "..", "payload"))
sys.path.insert(0, _SUB)

SAMPLE_SUBJECT = {
    "normalized_name": "OpenAI GPT-4o",
    "provider": "OpenAI",
    "release_date": "2024-05-13",
    "access_date": "2025-11-01",
    "harness": "default",
    "reasoning_effort": "",
    "harness_version": "",
    "subject_features_extra": "",
}
SAMPLE_ITEM = {
    "item_content": "What is the derivative of f(x) = x^3 * sin(x)?",
    "item_features": "tier=medium;points=1",
    "interactors": "",
}
SAMPLE_INPUT = [SAMPLE_SUBJECT, SAMPLE_ITEM]
SAMPLE_LABELED = [
    [[SAMPLE_SUBJECT, dict(SAMPLE_ITEM, item_content="Compute 17 * 24.")], 1],
    [[SAMPLE_SUBJECT, dict(SAMPLE_ITEM, item_content="Integrate x * e^(x^2).")], 0],
]


def _belief(probability):
    return {
        "p": probability,
        "confidence": "medium",
        "update_reasoning": "Scripted estimate for the walkthrough.",
    }


def _reply(name, arguments):
    """Minimal provider response used by the mock LLM and failure checks."""
    calls = (
        []
        if name is None
        else [
            types.SimpleNamespace(
                id="demo_call",
                function=types.SimpleNamespace(
                    name=name,
                    arguments=json.dumps(arguments),
                ),
            ),
        ]
    )
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    role="assistant",
                    content="",
                    tool_calls=calls,
                )
            )
        ],
        usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )


def _responses_reply(reply):
    """Script native Responses JSON, including state that must be replayed."""
    output = [
        {
            "type": "reasoning",
            "id": "rs_demo",
            "summary": [],
            "encrypted_content": "synthetic-encrypted-state",
        },
        {
            "type": "message",
            "id": "msg_demo",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "Inspecting the evidence.",
                    "annotations": [],
                }
            ],
        },
    ]
    for call in reply.choices[0].message.tool_calls or []:
        output.append(
            {
                "type": "function_call",
                "id": "fc_" + call.id,
                "call_id": call.id,
                "name": call.function.name,
                "arguments": call.function.arguments,
            }
        )
    return {
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": reply.usage.prompt_tokens,
            "output_tokens": reply.usage.completion_tokens,
            "total_tokens": reply.usage.prompt_tokens + reply.usage.completion_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 13},
        },
    }


@contextmanager
def _mock_llm(**kwargs):
    """Mock inference transport without importing optional provider SDKs."""
    mocked = Mock(**kwargs)

    def responses(endpoint, payload, timeout):
        assert endpoint == "responses"
        assert 0 < timeout <= 240
        return _responses_reply(mocked(**payload))

    with patch("agent.llm_client.post_json", side_effect=responses):
        yield mocked


def _search_reply():
    return _reply(
        "search_benchmarks",
        {
            "query": "math derivative calculus",
            "updated_belief": _belief(0.5),
        },
    )


def _submit_reply(probability=0.62):
    return _reply(
        "submit",
        {
            "probability": probability,
            "reasoning": "Scripted submission after inspecting the retrieved evidence.",
            "updated_belief": _belief(0.62),
        },
    )


def _install_mock_corpus(stack):
    """Use explicitly synthetic tables; do not read or download competition data."""
    import pandas as pd
    from comp_pipeline.script import databank, dataroot

    databank.get_db.cache_clear()
    stack.callback(databank.get_db.cache_clear)
    benchmarks = pd.DataFrame(
        [
            {
                "benchmark_id": "demo_math",
                "name": "Synthetic mathematics benchmark",
                "description": "Calculus examples for a code walkthrough; not measured results.",
                "domain": ["math"],
                "modality": ["text"],
                "release_date": "2025-01-01",
                "n_items": 1,
                "n_subjects": 1,
            }
        ]
    )
    subjects = pd.DataFrame(
        [
            {
                "subject_id": "demo_subject",
                "normalized_name": SAMPLE_SUBJECT["normalized_name"],
            }
        ]
    )
    tables = {"benchmarks": benchmarks, "subjects": subjects}
    stack.enter_context(
        patch.object(dataroot, "corpus_dirs", return_value=["demo_math"])
    )
    stack.enter_context(patch.object(dataroot, "emb_root", return_value=None))
    stack.enter_context(
        patch.object(
            databank,
            "_load",
            side_effect=lambda directory, table: tables[table].copy(),
        )
    )


def _local_payload():
    """Resolve data produced by prepare_data.py for a real prediction."""
    data = os.path.join(_PAYLOAD, "data")
    if not os.environ.get("BLF_DATA_ROOT") and os.path.isdir(data):
        with open(os.path.join(data, "corpus_manifest.json")) as handle:
            manifest = json.load(handle)
        if manifest.get("corpus_scope") != "paec-public-training":
            raise SystemExit(
                "payload/ predates the public release; prepare a new public payload"
            )
        os.environ["BLF_DATA_ROOT"] = data
    embeddings = os.path.join(_PAYLOAD, "embeddings")
    if not os.environ.get("BLF_EMB_ROOT") and os.path.isdir(embeddings):
        os.environ["BLF_EMB_ROOT"] = embeddings


def _check_proxy_retries():
    """Use urllib's real CONNECT handling; no external connection or API call."""
    from comp_pipeline.script import openai_api

    destinations = []

    class Proxy(BaseHTTPRequestHandler):
        def do_CONNECT(self):
            destinations.append(self.path)
            # Simulate a dropped TLS connection after accepting the tunnel.
            # Retrying a mutated Request would eventually ask for port 80.
            self.send_response(200 if self.path.endswith(":443") else 403)
            self.end_headers()
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    proxy = f"http://127.0.0.1:{server.server_port}"
    opener = build_opener(ProxyHandler({"https": proxy, "http": proxy}))
    try:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-test-key", "OPENAI_BASE_URL": "https://api.openai.com/v1"}),
            patch("urllib.request.proxy_bypass", return_value=False),
            patch.object(openai_api, "urlopen", side_effect=opener.open),
            patch.object(openai_api.time, "sleep"),
        ):
            try:
                openai_api.post_json("embeddings", {"input": ["synthetic query"]}, 60)
            except openai_api.OpenAIRequestError as exc:
                assert "TLS connection failed; attempts=3" in exc.safe_summary
            else:
                raise AssertionError("closed TLS tunnels unexpectedly succeeded")
        assert destinations == ["api.openai.com:443"] * 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _check_api_transport():
    """Check raw HTTP serialization, errors, and embeddings without SDKs."""
    from comp_pipeline.script import embedding_index, openai_api

    with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-test-key"}):
        with patch.object(
            openai_api, "urlopen", return_value=io.BytesIO(b'{"ok": true}')
        ) as send:
            assert openai_api.post_json("responses", {"model": "gpt-5.6-luna"}, 12) == {
                "ok": True
            }
            request = send.call_args.args[0]
            assert request.get_method() == "POST"
            assert request.get_header("Authorization") == "Bearer synthetic-test-key"
            assert json.loads(request.data) == {"model": "gpt-5.6-luna"}
            assert 0 < send.call_args.kwargs["timeout"] <= 12
            send.assert_called_once()

        error = HTTPError(
            "https://api.openai.com/v1/responses",
            401,
            "invalid credential",
            {},
            io.BytesIO(b'{"error": "synthetic-test-key: rate limited"}'),
        )
        with patch.object(openai_api, "urlopen", side_effect=error) as send:
            try:
                openai_api.post_json("responses", {}, 12)
            except RuntimeError as exc:
                assert "401" in str(exc)
                assert "synthetic-test-key" not in str(exc)
            else:
                raise AssertionError("HTTP failure did not propagate")
            send.assert_called_once()  # permanent errors are not retried

        for error in (
            HTTPError(
                "https://api.openai.com/v1/responses",
                429,
                "rate limited",
                {"Retry-After": "2"},
                io.BytesIO(),
            ),
            HTTPError(
                "https://api.openai.com/v1/responses",
                503,
                "unavailable",
                {},
                io.BytesIO(),
            ),
            URLError("temporary connection failure"),
        ):
            with (
                patch.object(
                    openai_api,
                    "urlopen",
                    side_effect=[error, io.BytesIO(b'{"ok": true}')],
                ) as send,
                patch.object(openai_api.time, "sleep") as sleep,
            ):
                assert openai_api.post_json("responses", {}, 12) == {"ok": True}
                assert send.call_count == 2
                sleep.assert_called_once()
                if isinstance(error, HTTPError) and error.code == 429:
                    sleep.assert_called_once_with(2)

        with (
            patch.object(
                openai_api, "urlopen", side_effect=URLError("unavailable")
            ) as send,
            patch.object(openai_api.time, "sleep"),
        ):
            _expect_error(
                lambda: openai_api.post_json("responses", {}, 12), RuntimeError
            )
            assert send.call_count == 3

        # Server backoff must not extend the caller's request budget.
        error = HTTPError(
            "https://api.openai.com/v1/responses",
            429,
            "rate limited",
            {"Retry-After": "60"},
            io.BytesIO(),
        )
        with (
            patch.object(openai_api, "urlopen", side_effect=error) as send,
            patch.object(openai_api.time, "sleep") as sleep,
        ):
            _expect_error(
                lambda: openai_api.post_json("responses", {}, 12), RuntimeError
            )
            send.assert_called_once()
            sleep.assert_not_called()

        # A slow Responses attempt may exhaust its 120-second socket timeout
        # while the prediction still has time to retry. Exercise the complete
        # call path with a simulated clock, including the exhausted-budget case.
        from agent.llm_client import complete_turn
        from agent.tools import SUBMIT_TOOL
        from config.config import AgentConfig

        def check_slow_turn(budget, should_recover):
            clock = [0.0]
            attempt_timeouts = []

            def slow_response(request, timeout):
                attempt_timeouts.append(timeout)
                payload = json.loads(request.data)
                assert payload["reasoning"] == {"effort": "medium"}
                if len(attempt_timeouts) == 1 or not should_recover:
                    clock[0] += timeout
                    raise TimeoutError("synthetic stalled response")
                clock[0] += 5
                return io.BytesIO(
                    json.dumps(_responses_reply(_submit_reply())).encode()
                )

            def advance_clock(seconds):
                clock[0] += seconds

            with (
                patch("agent.llm_client.post_json", openai_api.post_json),
                patch.object(openai_api, "urlopen", side_effect=slow_response),
                patch.object(
                    openai_api.time, "monotonic", side_effect=lambda: clock[0]
                ),
                patch.object(openai_api.time, "sleep", side_effect=advance_clock),
                patch.object(openai_api.random, "uniform", return_value=0),
            ):

                def request_turn():
                    return complete_turn(
                        [{"role": "user", "content": "Synthetic public test."}],
                        [SUBMIT_TOOL],
                        AgentConfig(),
                        budget,
                        True,
                    )

                if should_recover:
                    result = request_turn()
                    assert (
                        result.choices[0].message.tool_calls[0].function.name
                        == "submit"
                    )
                else:
                    _expect_error(request_turn, openai_api.OpenAIRequestError)
            assert attempt_timeouts == [120, budget - 121]
            assert clock[0] <= budget

        for budget, should_recover in ((200, True), (125, False)):
            check_slow_turn(budget, should_recover)

        # The failed BLF run used an embedding request with a 60-second budget.
        # A stalled first attempt must leave time for a fresh connection. Exercise
        # embed_query itself so the real embedding timeout and cache are covered.
        for recover in (True, False):
            clock = [0.0]
            timeouts = []
            slow_index = embedding_index.EmbeddingIndex(root="synthetic-index")

            def slow_embedding(request, timeout):
                timeouts.append(timeout)
                if len(timeouts) == 1 or not recover:
                    clock[0] += timeout
                    raise URLError(TimeoutError("synthetic stalled connection"))
                clock[0] += 0.5
                return io.BytesIO(b'{"data": [{"embedding": [3, 4, 0]}]}')

            def advance_embedding_clock(seconds):
                clock[0] += seconds

            with (
                patch.object(slow_index, "manifest", return_value={"model": "text-embedding-3-small", "dimensions": 3}),
                patch.object(openai_api, "urlopen", side_effect=slow_embedding),
                patch.object(openai_api.time, "monotonic", side_effect=lambda: clock[0]),
                patch.object(openai_api.time, "sleep", side_effect=advance_embedding_clock),
                patch.object(openai_api.random, "uniform", return_value=0),
            ):
                if recover:
                    vector = slow_index.embed_query("public synthetic query")
                    assert embedding_index.np.allclose(vector, [0.6, 0.8, 0])
                    assert slow_index.embed_query("public synthetic query") is vector
                    assert timeouts == [15, 15]
                else:
                    try:
                        slow_index.embed_query("public synthetic query")
                    except openai_api.OpenAIRequestError as exc:
                        assert "timeout; attempts=3" in str(exc)
                        assert str(exc) == exc.safe_summary
                    else:
                        raise AssertionError("persistent embedding failure did not raise")
                    assert timeouts == [15, 15, 15]
                assert clock[0] <= 60

        # Retain fixed network-cause categories without exposing error text.
        for reason, expected, attempts in (
            (socket.gaierror(-3, "sk_secretsecret hidden question"), "DNS lookup failed", 3),
            (ssl.SSLCertVerificationError(1, "sk_secretsecret"), "certificate verification failed", 1),
            (ConnectionResetError("sk_secretsecret"), "connection reset", 3),
            (OSError("Tunnel connection failed: 503 sk_secretsecret"), "proxy HTTP 503", 8),
            (OSError("Tunnel connection failed: 403 sk_secretsecret"), "proxy HTTP 403", 1),
        ):
            with (
                patch.object(openai_api, "urlopen", side_effect=URLError(reason)) as send,
                patch.object(openai_api.time, "sleep"),
            ):
                try:
                    openai_api.post_json("embeddings", {}, 60)
                except openai_api.OpenAIRequestError as exc:
                    assert expected in exc.safe_summary
                    assert "sk_secretsecret" not in str(exc)
                    assert "hidden question" not in str(exc)
                else:
                    raise AssertionError("connection failure did not raise")
                assert send.call_count == attempts

        # Replay the three ~30-second CONNECT failures seen in submission
        # 925683. A fourth attempt can recover without extending the turn budget
        # or resending a request that already reached the provider.
        for budget, recover in ((240, True), (90, False)):
            clock = [0.0]
            attempts = []

            def unavailable_proxy(request, timeout):
                attempts.append((request, timeout))
                if len(attempts) <= 3:
                    clock[0] += min(30, timeout)
                    raise URLError(OSError("Tunnel connection failed: 503 Service Unavailable"))
                clock[0] += 5
                return io.BytesIO(b'{"ok": true}')

            def advance_proxy_clock(seconds):
                clock[0] += seconds

            with (
                patch.object(openai_api, "urlopen", side_effect=unavailable_proxy),
                patch.object(openai_api.time, "monotonic", side_effect=lambda: clock[0]),
                patch.object(openai_api.time, "sleep", side_effect=advance_proxy_clock),
                patch.object(openai_api.random, "uniform", return_value=0),
            ):
                if recover:
                    assert openai_api.post_json("responses", {}, budget) == {"ok": True}
                    assert len(attempts) == 4
                    assert clock[0] == 102
                else:
                    _expect_error(lambda: openai_api.post_json("responses", {}, budget),
                                  openai_api.OpenAIRequestError)
                    assert len(attempts) == 3
            assert clock[0] <= budget
            assert len({id(request) for request, _ in attempts}) == len(attempts)

        # HTTP 503 from the API is distinct from a proxy CONNECT rejection.
        # Keep the original three-attempt limit for those requests.
        with (
            patch.object(openai_api, "urlopen", side_effect=[
                HTTPError("https://api.openai.com/v1/responses", 503, "unavailable", {}, io.BytesIO())
                for _ in range(3)
            ]) as send,
            patch.object(openai_api.time, "sleep"),
        ):
            _expect_error(lambda: openai_api.post_json("responses", {}, 240),
                          openai_api.OpenAIRequestError)
            assert send.call_count == 3

        index = embedding_index.EmbeddingIndex(root="synthetic-embedding-root")
        with (
            patch.object(
                index,
                "manifest",
                return_value={"model": "text-embedding-3-small", "dimensions": 3},
            ),
            patch.object(
                embedding_index,
                "post_json",
                return_value={"data": [{"embedding": [3, 4, 0]}]},
            ) as send,
        ):
            vector = index.embed_query("test query")
            assert embedding_index.np.allclose(vector, [0.6, 0.8, 0])
            assert index.embed_query("test query") is vector
            send.assert_called_once()
            assert send.call_args.args[0] == "embeddings"
            assert send.call_args.args[1]["encoding_format"] == "float"

    with (
        patch.dict(os.environ, {"OPENAI_API_KEY": ""}),
        patch.object(openai_api, "urlopen") as send,
    ):
        _expect_error(lambda: openai_api.post_json("responses", {}, 12), RuntimeError)
        send.assert_not_called()
    _check_proxy_retries()
    print(
        "[smoke] SDK-free HTTP requests, credential redaction, and query embeddings checked"
    )


def _show_walkthrough(result, mocked):
    if mocked:
        print("\nWalkthrough: synthetic evidence and scripted probabilities.")
    else:
        print("\nWalkthrough: one real prediction.")
    print("Initial p=0.50 is a placeholder; the LLM supplies subsequent estimates.")
    for entry, belief in zip(result["tool_log"], result["belief_history"][1:]):
        print(f"\nTurn {entry['step']}: {entry['tool']}, p={belief['p']:.2f}")
        print(f"  Why: {belief['update_reasoning']}")
        print(f"  Result: {entry['result'][:500]}")
        if entry["tool"] != "submit":
            print("  This result becomes evidence for the NEXT LLM turn.")
    print(f"\nSubmitted probability: {result['forecast']:.2f}")


def _expect_error(action, exception_type):
    try:
        action()
    except exception_type:
        return
    raise AssertionError(f"expected {exception_type.__name__}")


def _check_agent_failures(model):
    """Exercise the actual agent loop, including failures after a valid first turn."""
    from agent import agent
    from config.config import AgentConfig

    question = model.build_question(SAMPLE_SUBJECT, SAMPLE_ITEM)
    config = AgentConfig(max_steps=2)

    def run():
        return agent.run_agent(question, config)

    missing_probability = _reply(
        "submit",
        {
            "updated_belief": _belief(0.62),
            "reasoning": "Missing required probability",
        },
    )
    malformed = _search_reply()
    malformed.choices[0].message.tool_calls[0].function.arguments = "{"
    multiple = _search_reply()
    multiple.choices[0].message.tool_calls *= 2
    cases = [
        ([RuntimeError("first API call failed")], RuntimeError),
        ([_search_reply(), RuntimeError("second API call failed")], RuntimeError),
        ([_reply(None, {})], ValueError),
        ([multiple], ValueError),
        ([malformed], ValueError),
        ([_reply("unknown_tool", {})], ValueError),
        ([_search_reply(), _search_reply()], ValueError),  # last turn must submit
        ([missing_probability], KeyError),
        (
            [_reply("submit", {"probability": 0.62, "reasoning": "Missing belief"})],
            KeyError,
        ),
        (
            [
                _reply(
                    "submit",
                    {
                        "probability": 0.62,
                        "reasoning": "Missing belief p",
                        "updated_belief": {
                            "confidence": "medium",
                            "update_reasoning": "test",
                        },
                    },
                )
            ],
            KeyError,
        ),
        (
            [
                _reply(
                    "submit",
                    {
                        "probability": 0.62,
                        "reasoning": "Invalid evidence list",
                        "updated_belief": dict(
                            _belief(0.62), evidence_for="not a list"
                        ),
                    },
                )
            ],
            ValueError,
        ),
    ]
    for replies, exception_type in cases:
        with _mock_llm(side_effect=replies):
            _expect_error(run, exception_type)

    # Validate non-finite values BEFORE any clipping in both belief and submit.
    for value in (float("nan"), float("inf"), float("-inf")):
        bad_belief = _reply(
            "search_benchmarks",
            {
                "query": "math",
                "updated_belief": _belief(value),
            },
        )
        for reply in (bad_belief, _submit_reply(value)):
            with _mock_llm(return_value=reply):
                _expect_error(run, ValueError)

    for value in ("0.62", True, None):
        with _mock_llm(return_value=_submit_reply(value)):
            _expect_error(run, TypeError)

    from comp_pipeline.script.databank import MeasurementDB

    with (
        _mock_llm(return_value=_search_reply()),
        patch.object(
            MeasurementDB,
            "search_benchmarks",
            side_effect=OSError("unreadable corpus"),
        ),
    ):
        _expect_error(run, OSError)

    # Corrupt public tables and embedding API failures must also propagate.
    from comp_pipeline.script import databank, dataroot, embedding_index

    fresh_db = MeasurementDB(["demo_math"])
    with patch.object(databank, "_load", side_effect=OSError("bad table")):
        _expect_error(lambda: fresh_db.search_benchmarks("math"), OSError)
    with patch.object(embedding_index, "get_index") as get_index:
        get_index.return_value.search.side_effect = RuntimeError("embedding API failed")
        _expect_error(lambda: fresh_db.semantic_search_items("math"), RuntimeError)
    with patch.object(dataroot, "emb_root", return_value="synthetic-index"):
        tools = agent.get_tool_schemas()
        assert len(tools) == 9  # eight retrieval tools plus submit
        assert "semantic_search_items" in agent.get_system_prompt(2, tools)

    # A late submit must fail too; elapsed time is controlled without sleeping.
    with (
        _mock_llm(return_value=_submit_reply()),
        patch.object(
            agent,
            "time",
            types.SimpleNamespace(monotonic=Mock(side_effect=[0, 0, 241])),
        ),
    ):
        _expect_error(run, TimeoutError)

    for supplied, expected in ((-0.4, 0.02), (1.4, 0.98)):
        with _mock_llm(return_value=_submit_reply(supplied)):
            assert run()["forecast"] == expected

    with _mock_llm(return_value=_submit_reply()) as completion:
        one_turn = agent.run_agent(question, AgentConfig(max_steps=1))
        assert one_turn["n_steps"] == 1
        assert completion.call_args.kwargs["tool_choice"]["name"] == "submit"

    # Providers using Chat Completions retain their existing tool-call format.
    completion = Mock(return_value=_submit_reply())
    with (
        patch.dict(
            sys.modules, {"litellm": types.SimpleNamespace(completion=completion)}
        ),
        patch(
            "agent.llm_client.post_json",
            side_effect=AssertionError("unexpected endpoint"),
        ),
    ):
        assert (
            agent.run_agent(
                question, AgentConfig(llm="anthropic/claude-opus-4-8", max_steps=1)
            )["forecast"]
            == 0.62
        )
        assert (
            completion.call_args.kwargs["tool_choice"]["function"]["name"] == "submit"
        )

    # Check the public entry point as well as the loop's individual error paths.
    with patch.object(model, "run_agent", side_effect=RuntimeError("API failure")):
        _expect_error(lambda: model.predict(SAMPLE_INPUT, []), RuntimeError)
    with patch.object(
        model, "run_agent", return_value={"submitted": False, "forecast": 0.5}
    ):
        _expect_error(lambda: model.predict(SAMPLE_INPUT, []), RuntimeError)
    print(
        "[smoke] API/tool failures, invalid calls, and deadlines surface; bounds are consistent"
    )


def _check_api_preflight(check):
    from comp_pipeline.script.openai_api import OpenAIRequestError
    from config.config import AgentConfig

    config = AgentConfig(reasoning_effort="medium")
    index = Mock()
    index.manifest.return_value = {"model": "text-embedding-3-small", "dimensions": 2}
    with (
        patch("agent.llm_client.complete_turn", return_value=_submit_reply()) as turn,
        patch("comp_pipeline.script.embedding_index.get_index", return_value=index),
        patch(
            "comp_pipeline.script.api_preflight.post_json",
            return_value={"data": [{"embedding": [0.5, 0.5]}]},
        ) as embedding,
    ):
        check(config)
        assert turn.call_args.args[2] is config
        assert turn.call_args.kwargs["force_submit"] is True
        assert embedding.call_args.args[1]["dimensions"] == 2

    failure = OpenAIRequestError(
        "provider text containing a secret",
        "OpenAI responses: HTTP 401, code invalid_api_key",
    )
    with patch("agent.llm_client.complete_turn", side_effect=failure):
        try:
            check(config)
        except RuntimeError as exc:
            assert "HTTP 401" in str(exc)
            assert "provider text" not in str(exc) and "secret" not in str(exc)
            assert exc.__suppress_context__
        else:
            raise AssertionError("startup request failure did not propagate")
    print("[smoke] public startup probes preserve medium reasoning and redact failures")


def _check_acquisition(labeling):
    """Check decisions reuse the supplied prediction and honor the budget."""
    context = {"labels_remaining": 31, "labels_acquired": 0, "max_labels": 31,
               "items_remaining": 80, "subject_id": "subject_001", "benchmark_id": "benchmark_001"}
    decisions = [labeling.acquisition_function(SAMPLE_INPUT, p, SAMPLE_LABELED, context)
                 for p in (0.5, 0.25, 0.75, 0.0, 1.0)]
    assert decisions == [True, False, False, False, False]
    assert labeling.acquisition_function(SAMPLE_INPUT, 0.0, SAMPLE_LABELED,
                                         {**context, "items_remaining": 31}) is True
    assert labeling.acquisition_function(SAMPLE_INPUT, 0.5, SAMPLE_LABELED,
                                         {**context, "labels_remaining": 0}) is False
    assert isinstance(labeling.acquisition_function(SAMPLE_INPUT), float)
    assert not hasattr(labeling, "predict"), "Acquisition must reuse the supplied prediction"
    print("[smoke] streaming acquisition reuses predictions and respects remaining budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="synthetic corpus + scripted LLM; no API calls",
    )
    parser.add_argument(
        "--walkthrough",
        action="store_true",
        help="show the belief and evidence each turn",
    )
    args = parser.parse_args()

    with ExitStack() as stack:
        output_dir = stack.enter_context(
            tempfile.TemporaryDirectory(prefix="paec-smoke-")
        )
        stack.enter_context(patch.dict(os.environ, {"BLF_OUT": output_dir}))
        if not args.mock:
            _local_payload()
        else:
            from comp_pipeline.script.api_preflight import check_api_configuration

            stack.enter_context(
                patch("comp_pipeline.script.api_preflight.check_api_configuration")
            )

        import labeling
        import model

        requests = []
        if args.mock:
            _install_mock_corpus(stack)
            stack.enter_context(patch.dict(model.CFG, {"max_steps": 2}))

            def completion(**kwargs):
                requests.append(copy.deepcopy(kwargs))
                return _search_reply() if len(requests) % 2 else _submit_reply()

            stack.enter_context(_mock_llm(side_effect=completion))
            print("[smoke] synthetic corpus and LLM; no API calls")

        # Reuse predict()'s run for the walkthrough without another agent call.
        results = []
        original_run = model.run_agent

        def capture(*call_args, **kwargs):
            result = original_run(*call_args, **kwargs)
            results.append(result)
            return result

        with patch.object(model, "run_agent", side_effect=capture):
            probability = model.predict(SAMPLE_INPUT, SAMPLE_LABELED)
        assert isinstance(probability, float) and 0.02 <= probability <= 0.98
        assert results[0]["submitted"]
        print(f"[smoke] predict() -> {probability}")

        if args.walkthrough:
            _show_walkthrough(results[0], args.mock)

        # A different harness is a different subject even for the same model.
        other = dict(SAMPLE_SUBJECT, harness="different harness")
        assert model.build_question(other, SAMPLE_ITEM, SAMPLE_LABELED) == (
            model.build_question(other, SAMPLE_ITEM)
        )
        labeled_question = model.build_question(
            SAMPLE_SUBJECT, SAMPLE_ITEM, SAMPLE_LABELED
        )
        assert "- [CORRECT] Compute 17 * 24." in labeled_question["background"]
        assert "- [INCORRECT] Integrate x * e^(x^2)." in labeled_question["background"]
        mixed_labels = SAMPLE_LABELED + [[[other, SAMPLE_ITEM], 1]]
        assert model.build_question(SAMPLE_SUBJECT, SAMPLE_ITEM, mixed_labels) == (
            labeled_question
        )
        base = model.build_question(SAMPLE_SUBJECT, SAMPLE_ITEM)
        assert set(base) == {"question", "background", "resolution_criteria"}
        for field in ("item_content", "item_features", "interactors"):
            item = dict(SAMPLE_ITEM)
            item[field] += " changed"
            changed = model.build_question(SAMPLE_SUBJECT, item)
            assert changed["background"] != base["background"]

        if args.mock:
            assert probability == 0.62
            # Native output-token usage already includes reasoning tokens.
            assert results[0]["tokens_in"] == 200
            assert results[0]["tokens_out"] == 40
            assert [s["p"] for s in results[0]["belief_history"]] == [0.5, 0.5, 0.62]
            tool_messages = [
                m
                for m in requests[1]["input"]
                if m.get("type") == "function_call_output"
            ]
            assert "Synthetic mathematics benchmark" in tool_messages[0]["output"]
            assert "semantic_search_items" not in results[0]["system_prompt"]
            assert requests[0]["reasoning"] == {"effort": "medium"}
            assert requests[0]["store"] is False
            assert requests[0]["parallel_tool_calls"] is False
            assert "num_retries" not in requests[0]
            assert requests[0]["model"] == "gpt-5.6-luna"
            assert requests[0]["max_output_tokens"] == 32000
            assert requests[0]["tool_choice"] == "required"
            assert all(tool["strict"] is True for tool in requests[0]["tools"])
            for tool in requests[0]["tools"]:
                schema = tool["parameters"]
                assert schema["additionalProperties"] is False
                assert set(schema["required"]) == set(schema["properties"])
            previous_output = requests[1]["input"][2:5]
            assert [item["type"] for item in previous_output] == [
                "reasoning",
                "message",
                "function_call",
            ]
            assert (
                previous_output[0]["encrypted_content"] == "synthetic-encrypted-state"
            )
            assert (
                previous_output[1]["content"][0]["text"] == "Inspecting the evidence."
            )
            assert previous_output[2]["call_id"] == tool_messages[0]["call_id"]
            assert requests[1]["tool_choice"] == {"type": "function", "name": "submit"}
            assert [t["name"] for t in requests[1]["tools"]] == ["submit"]

            # An incomplete Responses result cannot provide a final prediction.
            from agent import agent
            from config.config import AgentConfig

            # Nullable optional strict-schema fields keep tool defaults and
            # prior belief fields; native output remains unchanged for replay.
            nullable = _search_reply()
            call = nullable.choices[0].message.tool_calls[0]
            arguments = json.loads(call.function.arguments)
            arguments["updated_belief"]["evidence_for"] = None
            arguments["updated_belief"]["base_rate_anchor"] = None
            call.function.arguments = json.dumps(arguments)
            with _mock_llm(side_effect=[nullable, _submit_reply()]):
                assert (
                    agent.run_agent(base, AgentConfig(max_steps=2))["forecast"] == 0.62
                )

            incomplete = _responses_reply(_submit_reply())
            incomplete["status"] = "incomplete"
            with patch("agent.llm_client.post_json", return_value=incomplete):
                _expect_error(
                    lambda: agent.run_agent(base, AgentConfig()), RuntimeError
                )

            # Token-limited turns can retry with more output space, but their
            # partial submit calls must never become a prediction or history.
            limited = _responses_reply(_submit_reply())
            limited.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
            limited["output"][-1]["arguments"] = '{"probability":'
            limited["usage"].update(output_tokens=32000, total_tokens=32100)
            with patch("agent.llm_client.post_json", side_effect=[limited, _responses_reply(_submit_reply())]) as api:
                recovered = agent.run_agent(base, AgentConfig(max_steps=1))
            assert recovered["forecast"] == 0.62
            assert recovered["tokens_in"] == 200 and recovered["tokens_out"] == 32020
            first, second = [call.args[1] for call in api.call_args_list]
            assert first["max_output_tokens"] == 32000 and second["max_output_tokens"] == 64000
            assert {k: v for k, v in first.items() if k != "max_output_tokens"} == {
                k: v for k, v in second.items() if k != "max_output_tokens"
            }
            assert second["reasoning"] == {"effort": "medium"}
            assert second["tool_choice"] == {"type": "function", "name": "submit"}
            assert 0 < api.call_args_list[1].kwargs["timeout"] <= api.call_args_list[0].kwargs["timeout"]

            # Persistent truncation remains a failure after bounded retries.
            with patch("agent.llm_client.post_json", return_value=limited) as api:
                _expect_error(lambda: agent.run_agent(base, AgentConfig(max_steps=1)), RuntimeError)
            assert [call.args[1]["max_output_tokens"] for call in api.call_args_list] == [32000, 64000, 128000]

            # Refusals/unknown incompletions are not retried as token limits,
            # and free-form provider fields never become visible diagnostics.
            from agent.llm_client import _incomplete_response_diagnostic
            for reason in ("content_filter", "private-sentinel-token"):
                rejected = {**limited, "incomplete_details": {"reason": reason}}
                with patch("agent.llm_client.post_json", return_value=rejected) as api:
                    _expect_error(lambda: agent.run_agent(base, AgentConfig(max_steps=1)), RuntimeError)
                assert api.call_count == 1
                assert "private-sentinel-token" not in _incomplete_response_diagnostic(rejected, 1)

            elapsed = [0.0]
            def consume_deadline(*args, **kwargs):
                elapsed[0] += 241
                return limited
            with (
                patch("agent.llm_client.time", types.SimpleNamespace(monotonic=lambda: elapsed[0])),
                patch("agent.llm_client.post_json", side_effect=consume_deadline) as api,
            ):
                _expect_error(lambda: agent.run_agent(base, AgentConfig(max_steps=1)), TimeoutError)
            assert api.call_count == 1

            # The predictor setting reaches the Responses request independently
            # of the target subject's reasoning_effort attribute.
            with (
                patch.dict(model.CFG, {"reasoning_effort": "high"}),
                _mock_llm(return_value=_submit_reply()) as configured,
            ):
                assert model.predict(SAMPLE_INPUT, SAMPLE_LABELED) == 0.62
                assert configured.call_args.kwargs["reasoning"] == {"effort": "high"}

            # Identical inputs must still run again and keep separate evidence
            # files. The public corpus is passed in memory, without split files.
            with patch.object(model, "run_agent", side_effect=capture):
                assert model.predict(SAMPLE_INPUT, SAMPLE_LABELED) == probability
            assert results[0]["run_id"] != results[1]["run_id"]
            assert results[0]["question_prompt"] == results[1]["question_prompt"]
            assert not any(
                item.get("type") == "reasoning" for item in requests[2]["input"]
            )
            for result in results:
                evidence_file = (
                    Path(result["log_dir"]) / result["tool_log"][0]["result_file"]
                )
                assert "Synthetic mathematics benchmark" in evidence_file.read_text()
            assert not list(Path(output_dir).rglob("*.json"))
            print(
                "[smoke] repeated inputs use separate logs; no corpus-configuration files"
            )

            before = len(requests)
            assert model.predict(SAMPLE_LABELED[0][0], SAMPLE_LABELED) == 0.62
            assert len(requests) == before + 2, "revealed label bypassed reasoning"
            _check_agent_failures(model)
            _check_api_transport()
            _check_api_preflight(check_api_configuration)

        # The platform passes the prediction already computed for this candidate.
        _check_acquisition(labeling)

        print(
            "[smoke] subject isolation, all item fields, and revealed evidence checked"
        )
        print("[smoke] OK")


if __name__ == "__main__":
    main()
