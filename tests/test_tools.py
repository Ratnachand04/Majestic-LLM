"""Tests for real tools/experts: web (mocked HTTP), code exec, specialist."""
from __future__ import annotations

import httpx
import pytest

from majestic.experts.specialist import SpecialistExpert
from majestic.experts.tools import CodeExecTool, WebTool, _strip_html


# --- WebTool (offline via httpx.MockTransport) -------------------------- #
def _mock_web_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if "duckduckgo" in request.url.host:
            return httpx.Response(200, json={"AbstractText": "Paris is the capital."})
        return httpx.Response(
            200, text="<html><body><h1>Title</h1><p>Body text</p><script>x=1</script></body></html>"
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_web_search_parses_abstract():
    web = WebTool(client=_mock_web_client())
    assert web.run(query="capital of France") == "Paris is the capital."


def test_web_fetch_strips_html():
    web = WebTool(client=_mock_web_client())
    out = web.run(url="https://example.com")
    assert "Title" in out and "Body text" in out
    assert "<" not in out and "x=1" not in out


def test_web_requires_argument():
    web = WebTool(client=_mock_web_client())
    with pytest.raises(ValueError):
        web.run()


def test_strip_html_removes_scripts_and_tags():
    assert _strip_html("<p>Hello <b>world</b></p><style>a{}</style>") == "Hello world"


# --- CodeExecTool (real subprocess) ------------------------------------- #
def test_code_exec_runs_and_captures_stdout():
    tool = CodeExecTool(timeout=15)
    result = tool.run(code="print(6 * 7)")
    assert result["stdout"] == "42"
    assert result["returncode"] == 0
    assert result["timed_out"] is False


def test_code_exec_reports_errors():
    tool = CodeExecTool(timeout=15)
    result = tool.run(code="raise ValueError('boom')")
    assert result["returncode"] != 0
    assert "ValueError" in result["stderr"]


def test_code_exec_times_out():
    tool = CodeExecTool(timeout=1)
    result = tool.run(code="import time; time.sleep(5)")
    assert result["timed_out"] is True


# --- SpecialistExpert (heuristic backend) ------------------------------- #
def test_specialist_positive():
    out = SpecialistExpert().run(text="This is great, I love it")
    assert out["label"] == "positive"
    assert out["backend"] == "heuristic"


def test_specialist_negative():
    out = SpecialistExpert().run(text="terrible and awful, i hate it")
    assert out["label"] == "negative"


def test_specialist_neutral():
    out = SpecialistExpert().run(text="the object is on the table")
    assert out["label"] == "neutral"
