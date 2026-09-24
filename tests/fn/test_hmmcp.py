"""Tests for hmmcp.geron_model_context_protocol."""

import doctest as _doctest

from morie.fn import _array_core as np

from morie.fn.hmmcp import geron_model_context_protocol


def test_hmmcp_basic():
    """Test basic functionality."""
    TOOLS = [{"name": "add", "description": "add two numbers"}]

    def server(req):
        m = req["method"]
        if m == "tools/list":
            return {"jsonrpc": "2.0", "id": req["id"], "result": {"tools": TOOLS}}
        if m == "tools/call":
            a = req["params"]["arguments"]["a"]
            b = req["params"]["arguments"]["b"]
            return {"jsonrpc": "2.0", "id": req["id"], "result": {"content": a + b}}
        return {"jsonrpc": "2.0", "id": req["id"],
                "error": {"code": -32601, "message": "Method not found"}}

    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "add", "arguments": {"a": 2, "b": 3}}},
    ]

    result = geron_model_context_protocol(server, reqs)

    assert isinstance(result, dict)
    assert "exchanges" in result
    assert "n_ok" in result
    assert "n_errors" in result
    assert "methods" in result
    assert "wire_bytes" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert len(result["exchanges"]) == 2
    assert result["n_ok"] == 2
    assert result["n_errors"] == 0
    assert result["exchanges"][1]["response"]["result"]["content"] == 5


def test_hmmcp_edge():
    """Test edge cases - unknown method returns a well-formed error."""
    def server(req):
        return {"jsonrpc": "2.0", "id": req["id"],
                "error": {"code": -32601, "message": "Method not found"}}

    reqs = [{"jsonrpc": "2.0", "id": 9, "method": "nope"}]

    result = geron_model_context_protocol(server, reqs)

    assert isinstance(result, dict)
    assert "n_ok" in result
    assert "n_errors" in result
    assert "exchanges" in result
    assert result["n_ok"] == 0
    assert result["n_errors"] == 1
    assert result["exchanges"][0]["error_name"] == "Method not found"


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import morie.fn.hmmcp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
