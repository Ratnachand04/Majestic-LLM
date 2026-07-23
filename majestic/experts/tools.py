"""Real tools the router can dispatch to: web access and code execution.

Both are written so they can be exercised in tests without a live network:
``WebTool`` accepts an injectable HTTP client (use ``httpx.MockTransport``), and
``CodeExecTool`` runs the local Python interpreter, which is always available.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from typing import Any

from majestic.experts.base import Expert
from majestic.logging_utils import get_logger

logger = get_logger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    """Crudely reduce an HTML document to readable text."""
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


class WebTool(Expert):
    """Search the web and fetch pages.

    Parameters
    ----------
    client:
        An optional ``httpx.Client`` (inject an ``httpx.MockTransport`` in tests).
        When ``None`` a real client is created lazily per call.
    search_url:
        A JSON search endpoint. Defaults to the DuckDuckGo Instant Answer API.
    """

    name = "web"
    capabilities = ("search", "browse", "scrape")

    def __init__(
        self,
        client: Any | None = None,
        search_url: str = "https://api.duckduckgo.com/",
        timeout: float = 10.0,
        max_chars: int = 2000,
    ) -> None:
        self._client = client
        self.search_url = search_url
        self.timeout = timeout
        self.max_chars = max_chars

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import httpx  # lazy: only needed when actually hitting the network

        return httpx.Client(timeout=self.timeout, follow_redirects=True)

    def run(self, **kwargs: Any) -> Any:
        url = kwargs.get("url")
        query = kwargs.get("query") or kwargs.get("text")
        if url:
            return self._fetch(str(url))
        if query:
            return self._search(str(query))
        raise ValueError("WebTool.run requires 'url' or 'query'/'text'")

    def _fetch(self, url: str) -> str:
        resp = self._get_client().get(url)
        resp.raise_for_status()
        return _strip_html(resp.text)[: self.max_chars]

    def _search(self, query: str) -> str:
        resp = self._get_client().get(
            self.search_url,
            params={"q": query, "format": "json", "no_html": 1, "no_redirect": 1},
        )
        resp.raise_for_status()
        data = resp.json()
        abstract = data.get("AbstractText") or data.get("Answer")
        if abstract:
            return str(abstract)
        topics = data.get("RelatedTopics", [])
        texts = [
            t["Text"] for t in topics if isinstance(t, dict) and t.get("Text")
        ]
        return " | ".join(texts[:3]) if texts else f"No results for {query!r}"

    def estimate_cost(self, **kwargs: Any) -> float:
        return 3.0  # network round-trip is the most expensive path


class CodeExecTool(Expert):
    """Execute a snippet of Python in an isolated subprocess with a timeout.

    Isolation uses ``python -I`` (ignore env vars and user site) in a throwaway
    working directory. This is *not* a security sandbox — it constrains
    accidents, not adversaries — so never run untrusted code with it in
    production without a real sandbox (container / seccomp / nsjail).
    """

    name = "code_exec"
    capabilities = ("run_code",)

    def __init__(self, timeout: float = 5.0, python_executable: str | None = None) -> None:
        self.timeout = timeout
        self.python_executable = python_executable or sys.executable

    def run(self, **kwargs: Any) -> dict[str, Any]:
        code = kwargs.get("code") or kwargs.get("text") or ""
        return self._exec(str(code))

    def _exec(self, code: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as cwd:
            try:
                proc = subprocess.run(
                    [self.python_executable, "-I", "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=cwd,
                )
                return {
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                    "returncode": proc.returncode,
                    "timed_out": False,
                }
            except subprocess.TimeoutExpired as exc:
                logger.warning("code execution timed out after %ss", self.timeout)
                return {
                    "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
                    "stderr": "timeout",
                    "returncode": None,
                    "timed_out": True,
                }

    def estimate_cost(self, **kwargs: Any) -> float:
        return 2.0
