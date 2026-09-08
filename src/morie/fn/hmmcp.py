# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Model Context Protocol (MCP) for LLM tool integration."""

import json

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_model_context_protocol"]

_METHOD = "MCP / JSON-RPC 2.0 exchange with envelope validation"

_JSONRPC = "2.0"
_ERROR_CODES = {
    -32700: "Parse error",
    -32600: "Invalid Request",
    -32601: "Method not found",
    -32602: "Invalid params",
    -32603: "Internal error",
}


def _validate_request(req, index):
    if not isinstance(req, dict):
        raise ValueError(f"geron_model_context_protocol: request {index} is not a JSON object")
    if req.get("jsonrpc") != _JSONRPC:
        raise ValueError(
            f"geron_model_context_protocol: request {index} has jsonrpc={req.get('jsonrpc')!r}, expected '2.0'"
        )
    if "method" not in req or not isinstance(req["method"], str):
        raise ValueError(f"geron_model_context_protocol: request {index} has no string 'method'")
    if "id" in req and not isinstance(req["id"], (str, int)):
        raise ValueError(f"geron_model_context_protocol: request {index} has a non-scalar id {req['id']!r}")
    if "params" in req and not isinstance(req["params"], (dict, list)):
        raise ValueError(
            f"geron_model_context_protocol: request {index} has params of type "
            f"{type(req['params']).__name__}; JSON-RPC allows only object or array"
        )
    return req


def _validate_response(resp, req, index):
    if not isinstance(resp, dict):
        raise ValueError(f"geron_model_context_protocol: response {index} is not a JSON object")
    if resp.get("jsonrpc") != _JSONRPC:
        raise ValueError(
            f"geron_model_context_protocol: response {index} has jsonrpc={resp.get('jsonrpc')!r}, expected '2.0'"
        )
    has_result = "result" in resp
    has_error = "error" in resp
    if has_result == has_error:
        raise ValueError(
            f"geron_model_context_protocol: response {index} must carry exactly one of 'result' or 'error', "
            f"got {'both' if has_result else 'neither'}"
        )
    if resp.get("id") != req.get("id"):
        raise ValueError(
            f"geron_model_context_protocol: response {index} has id {resp.get('id')!r} but the request had "
            f"{req.get('id')!r}; a mismatched id is how a client attributes one tool's answer to another"
        )
    if has_error:
        err = resp["error"]
        if not isinstance(err, dict) or "code" not in err or "message" not in err:
            raise ValueError(
                f"geron_model_context_protocol: response {index} has a malformed error object; "
                f"'code' and 'message' are required"
            )
        if not isinstance(err["code"], int):
            raise ValueError(f"geron_model_context_protocol: response {index} error code must be an integer")
    return resp


def geron_model_context_protocol(server, client, requests=None):
    """
    Model Context Protocol (MCP) for LLM tool integration.

    Formula: JSON-RPC schema for exposing tools/resources to LLMs

    MCP is JSON-RPC 2.0 with an agreed method vocabulary
    (``initialize``, ``tools/list``, ``tools/call``, ``resources/read``),
    so the checkable content of "implement MCP" is the *envelope*, and
    that is what this validates on a real exchange:

    * every request carries ``jsonrpc: "2.0"`` and a string ``method``;
    * ``params``, if present, is an object or an array -- never a bare
      scalar;
    * every response carries exactly one of ``result`` or ``error``
      (both, or neither, is malformed);
    * the response ``id`` equals the request ``id``.  This is the check
      that matters most in practice: with concurrent tool calls in
      flight, a mismatched id is how a client attributes one tool's
      answer to a different tool's question.
    * every message round-trips through ``json.dumps``/``loads``, since
      a payload that cannot be serialised is not a protocol message
      however well-shaped it looks in memory.

    ``client`` supplies the requests -- a callable returning them, or the
    ``requests`` argument -- and ``server`` answers them:
    ``server(request_dict) -> response_dict``.

    Parameters
    ----------
    server : callable
        ``server(request) -> response``.
    client : callable or sequence
        Callable returning a list of requests, or the requests directly.
    requests : sequence, optional
        Requests, overriding ``client``.

    Returns
    -------
    result : RichResult
        Keys: exchanges, n_ok, n_errors, methods, wire_bytes,
        estimate, n, method.

    Examples
    --------
    A two-call session: list the tools, then call one.

    >>> TOOLS = [{"name": "add", "description": "add two numbers"}]
    >>> def server(req):
    ...     m = req["method"]
    ...     if m == "tools/list":
    ...         return {"jsonrpc": "2.0", "id": req["id"], "result": {"tools": TOOLS}}
    ...     if m == "tools/call":
    ...         a, b = req["params"]["arguments"]["a"], req["params"]["arguments"]["b"]
    ...         return {"jsonrpc": "2.0", "id": req["id"], "result": {"content": a + b}}
    ...     return {"jsonrpc": "2.0", "id": req["id"],
    ...             "error": {"code": -32601, "message": "Method not found"}}
    >>> reqs = [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    ...         {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
    ...          "params": {"name": "add", "arguments": {"a": 2, "b": 3}}}]
    >>> r = geron_model_context_protocol(server, reqs)
    >>> r["n_ok"], r["n_errors"]
    (2, 0)
    >>> r["exchanges"][1]["response"]["result"]["content"]
    5

    An unknown method comes back as a well-formed error, and that is
    counted, not raised:

    >>> e = geron_model_context_protocol(server, [{"jsonrpc": "2.0", "id": 9, "method": "nope"}])
    >>> e["n_ok"], e["n_errors"], e["exchanges"][0]["error_name"]
    (0, 1, 'Method not found')

    A server that answers the wrong id is caught:

    >>> bad = lambda req: {"jsonrpc": "2.0", "id": 999, "result": {}}
    >>> geron_model_context_protocol(bad, [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}])
    Traceback (most recent call last):
        ...
    ValueError: geron_model_context_protocol: response 0 has id 999 but the request had 1; a mismatched id is how a client attributes one tool's answer to another

    A response carrying both result and error is malformed:

    >>> both = lambda req: {"jsonrpc": "2.0", "id": req["id"], "result": {},
    ...                     "error": {"code": -1, "message": "x"}}
    >>> geron_model_context_protocol(both, [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}])
    Traceback (most recent call last):
        ...
    ValueError: geron_model_context_protocol: response 0 must carry exactly one of 'result' or 'error', got both

    References
    ----------
    Géron Ch 15
    """
    if not callable(server):
        raise ValueError(f"geron_model_context_protocol: server must be callable, got {type(server).__name__}")
    if requests is not None:
        reqs = list(requests)
    elif callable(client):
        reqs = list(client())
    else:
        try:
            reqs = list(client)
        except TypeError:
            raise ValueError(
                "geron_model_context_protocol: client must be a callable returning requests, or a sequence of them"
            ) from None
    if not reqs:
        raise ValueError("geron_model_context_protocol: no requests to exchange")

    exchanges = []
    n_ok = n_err = 0
    wire = 0
    methods = []
    for i, raw in enumerate(reqs):
        req = _validate_request(raw, i)
        try:
            req_wire = json.dumps(req)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"geron_model_context_protocol: request {i} is not JSON-serialisable: {exc}") from None
        resp = server(json.loads(req_wire))
        resp = _validate_response(resp, req, i)
        try:
            resp_wire = json.dumps(resp)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"geron_model_context_protocol: response {i} is not JSON-serialisable: {exc}") from None
        wire += len(req_wire) + len(resp_wire)
        methods.append(req["method"])
        is_err = "error" in resp
        if is_err:
            n_err += 1
        else:
            n_ok += 1
        exchanges.append(
            {
                "request": req,
                "response": json.loads(resp_wire),
                "method": req["method"],
                "ok": not is_err,
                "error_name": _ERROR_CODES.get(resp["error"]["code"], resp["error"]["message"]) if is_err else None,
            }
        )

    return RichResult(
        title="MCP exchange",
        summary_lines=[
            ("Exchanges", len(exchanges)),
            ("Successful", n_ok),
            ("Errors", n_err),
            ("Wire bytes", wire),
        ],
        interpretation=(
            "The id correlation is the load-bearing check: with concurrent tool calls, a mismatched "
            "id silently attributes one tool's answer to another's question."
        ),
        payload={
            "exchanges": exchanges,
            "n_ok": n_ok,
            "n_errors": n_err,
            "methods": methods,
            "wire_bytes": wire,
            "estimate": float(n_ok),
            "n": len(exchanges),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmmcp: MCP/JSON-RPC 2.0 exchange -- envelope, id correlation and serialisability all enforced"
