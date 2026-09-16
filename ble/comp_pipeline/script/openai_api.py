"""OpenAI JSON requests using only Python's standard library.

The competition does not install optional inference SDKs. Both Luna predictions
and query embeddings use this transport. Temporary failures receive bounded
retries within the caller's existing time budget.
"""

import json
import os
import random
import re
import socket
import ssl
import time
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_MAX_ATTEMPTS = 3
_MAX_PROXY_ATTEMPTS = 8
_MAX_ATTEMPT_SECONDS = 120
_EMBEDDING_ATTEMPT_SECONDS = 15
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class OpenAIRequestError(RuntimeError):
    """A request failure with a summary that contains no input or credentials."""

    def __init__(self, message, safe_summary):
        super().__init__(message)
        self.safe_summary = safe_summary


def _connection_failure(exc):
    """Classify transport failures without copying messages, URLs, or secrets."""
    reason = exc.reason if isinstance(exc, URLError) else exc
    if isinstance(reason, ssl.SSLCertVerificationError):
        return "certificate verification failed", False
    if isinstance(reason, TimeoutError):
        return "timeout", True
    if isinstance(reason, socket.gaierror):
        return "DNS lookup failed", True
    if isinstance(reason, ssl.SSLError):
        return "TLS connection failed", True
    if isinstance(reason, ConnectionResetError):
        return "connection reset", True
    if isinstance(reason, ConnectionRefusedError):
        return "connection refused", True
    # urllib wraps an HTTP CONNECT proxy rejection in URLError(OSError(...)).
    # Only recognize fixed HTTP statuses; the rest of the proxy text stays private.
    proxy = re.match(r"Tunnel connection failed: (\d{3})\b", str(reason))
    if proxy:
        status = int(proxy.group(1))
        if status in {400, 401, 403, 407, 408, 429, 500, 502, 503, 504}:
            return f"proxy HTTP {status}", status in _RETRYABLE_STATUS
    return "network connection failed", True


def _retry_delay(headers, attempt):
    """Honor server backoff hints, otherwise use exponential backoff + jitter."""
    for name, divisor in (("retry-after-ms", 1000), ("Retry-After", 1)):
        value = headers.get(name) if headers else None
        if value is not None:
            try:
                delay = float(value) / divisor
            except ValueError:
                try:
                    delay = parsedate_to_datetime(value).timestamp() - time.time()
                except (TypeError, ValueError, OverflowError):
                    continue
            if 0 <= delay < float("inf"):
                return delay
    return 2**attempt + random.uniform(0, 0.5)


def post_json(endpoint: str, payload: dict, timeout: float) -> dict:
    """Retry within the total budget, capping each attempt separately.

    Embedding attempts use a shorter socket timeout so their 60-second budget
    leaves room for retries. Responses retain the longer reasoning allowance.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    base = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for variable, header in (
        ("OPENAI_ORG_ID", "OpenAI-Organization"),
        ("OPENAI_PROJECT_ID", "OpenAI-Project"),
    ):
        if os.environ.get(variable):
            headers[header] = os.environ[variable]
    url = f"{base.rstrip('/')}/{endpoint}"
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    deadline = time.monotonic() + timeout
    attempt_limit = (
        _EMBEDDING_ATTEMPT_SECONDS if endpoint == "embeddings" else _MAX_ATTEMPT_SECONDS
    )
    request_failures = 0
    for attempt in range(_MAX_PROXY_ATTEMPTS):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"OpenAI {endpoint} exhausted its request time budget")
        # ProxyHandler mutates Request.host, type and _tunnel_host. Reusing a
        # proxied HTTPS request can turn a later attempt into CONNECT on port 80.
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(
                request, timeout=min(attempt_limit, remaining)
            ) as response:
                return json.load(response)
        except HTTPError as exc:
            request_failures += 1
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                code = json.loads(detail).get("error", {}).get("code")
            except (ValueError, AttributeError, TypeError):
                code = None
            if not isinstance(code, str) or not re.fullmatch(r"[a-z_]{1,64}", code):
                code = "unspecified"
            # Error responses can quote an invalid key. Keep credentials out of logs.
            detail = detail.replace(key, "[REDACTED]")[:1200]
            error = OpenAIRequestError(
                f"OpenAI {endpoint} failed: HTTP {exc.code}: {detail}",
                f"OpenAI {endpoint}: HTTP {exc.code}, code {code}",
            )
            retryable = exc.code in _RETRYABLE_STATUS
            delay = _retry_delay(exc.headers, attempt)
            exc.close()
        except (URLError, TimeoutError, ConnectionError) as exc:
            cause, retryable = _connection_failure(exc)
            # A rejected CONNECT tunnel has not sent the HTTPS API request.
            # Allow more recovery attempts for a temporary proxy outage without
            # increasing retries for requests that might have reached the model.
            proxy_unavailable = retryable and cause.startswith("proxy HTTP ")
            if not proxy_unavailable:
                request_failures += 1
            summary = (
                f"OpenAI {endpoint} connection failed ({cause}; "
                f"attempts={attempt + 1})"
            )
            error = OpenAIRequestError(
                summary,
                summary,
            )
            delay = _retry_delay(None, min(attempt, 3) if proxy_unavailable else attempt)
        if (not retryable or request_failures >= _MAX_ATTEMPTS
                or attempt + 1 == _MAX_PROXY_ATTEMPTS):
            raise error from None
        if delay >= deadline - time.monotonic():
            raise error from None
        time.sleep(delay)
