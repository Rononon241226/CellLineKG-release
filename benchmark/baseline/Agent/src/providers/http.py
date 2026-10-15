"""Send JSON API requests without requiring provider SDK packages."""

from __future__ import annotations

import json
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from benchmark.baseline.Agent.src.providers.base import ProviderError


class JsonTransport(Protocol):
    """Describe the injectable HTTP transport used by provider adapters."""

    def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Send one JSON POST request.

        Arg:
            url: Complete API endpoint.
            headers: HTTP request headers.
            payload: JSON-compatible request body.
            timeout_seconds: Request timeout in seconds.

        Returns:
            Parsed JSON response object.
        """
        ...


class UrllibJsonTransport:
    """Implement JSON POST requests with the Python standard library."""

    def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Send and decode one provider request.

        Arg:
            url: Complete API endpoint.
            headers: HTTP request headers.
            payload: JSON-compatible request body.
            timeout_seconds: Request timeout in seconds.

        Returns:
            Parsed provider response.
        """
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers = {"Content-Type": "application/json", **dict(headers)}
        request = Request(url, data=body, headers=request_headers, method="POST")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(
                f"Provider returned HTTP {exc.code}: {error_body}"
            ) from exc
        except URLError as exc:
            raise ProviderError(f"Provider request failed: {exc.reason}") from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise ProviderError("Provider returned invalid JSON.") from exc
        if not isinstance(parsed, dict):
            raise ProviderError("Provider response must be a JSON object.")
        return parsed
