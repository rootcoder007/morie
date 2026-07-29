# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Computational graph for expression differentiation."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_computational_graph"]

_BINARY = {"add", "sub", "mul", "div", "pow"}
_UNARY = {"neg", "exp", "log", "sin", "cos", "tanh", "sqrt", "square"}


def _forward(node, env, nodes, cache):
    key = id(node)
    if key in cache:
        return cache[key]
    if isinstance(node, (int, float, np.floating, np.integer)):
        v = float(node)
        idx = len(nodes)
        nodes.append({"op": "const", "value": v, "inputs": []})
    elif isinstance(node, str):
        if node not in env:
            raise ValueError(f"geron_computational_graph: variable {node!r} has no value in `values`")
        v = float(env[node])
        idx = len(nodes)
        nodes.append({"op": "var", "name": node, "value": v, "inputs": []})
    elif isinstance(node, (tuple, list)) and node:
        op = node[0]
        if not isinstance(op, str):
            raise ValueError(f"geron_computational_graph: the first element of a node must be an op name, got {op!r}")
        args = [_forward(a, env, nodes, cache) for a in node[1:]]
        vals = [nodes[i]["value"] for i in args]
        if op in _BINARY:
            if len(args) != 2:
                raise ValueError(f"geron_computational_graph: op {op!r} takes 2 inputs, got {len(args)}")
            a, b = vals
            if op == "add":
                v = a + b
            elif op == "sub":
                v = a - b
            elif op == "mul":
                v = a * b
            elif op == "div":
                if b == 0:
                    raise ValueError("geron_computational_graph: division by zero")
                v = a / b
            else:
                if a <= 0 and float(b) != int(b):
                    raise ValueError(f"geron_computational_graph: pow({a}, {b}) is not real-differentiable")
                v = a**b
        elif op in _UNARY:
            if len(args) != 1:
                raise ValueError(f"geron_computational_graph: op {op!r} takes 1 input, got {len(args)}")
            a = vals[0]
            if op == "neg":
                v = -a
            elif op == "exp":
                v = math.exp(a)
            elif op == "log":
                if a <= 0:
                    raise ValueError(f"geron_computational_graph: log requires a positive argument, got {a}")
                v = math.log(a)
            elif op == "sin":
                v = math.sin(a)
            elif op == "cos":
                v = math.cos(a)
            elif op == "tanh":
                v = math.tanh(a)
            elif op == "sqrt":
                if a <= 0:
                    raise ValueError(f"geron_computational_graph: sqrt is not differentiable at {a}")
                v = math.sqrt(a)
            else:
                v = a * a
        else:
            raise ValueError(
                f"geron_computational_graph: unknown op {op!r}; supported: {sorted(_BINARY | _UNARY)}"
            )
        idx = len(nodes)
        nodes.append({"op": op, "value": v, "inputs": args})
    else:
        raise ValueError(f"geron_computational_graph: cannot interpret node {node!r}")
    cache[key] = idx
    return idx


def geron_computational_graph(expr, values=None):
    """
    Computational graph for expression differentiation.

    Formula: DAG of ops with forward and backward pass

    An expression is a nested tuple ``(op, arg, ...)`` whose leaves are
    numbers or variable names, e.g.
    ``("add", ("mul", "x", "y"), ("exp", "x"))``. It is flattened into a
    DAG in topological order (shared sub-expressions are visited once and
    reused, which is the whole reason a graph beats a tree), evaluated
    forward, then differentiated by one reverse sweep -- so every partial
    derivative costs one pass in total, not one pass per variable.

    ``grad`` holds the derivative with respect to each named variable, and
    ``node_grads`` the adjoint of every intermediate, which is what a
    debugger of a real autodiff engine actually needs.

    Ops: add, sub, mul, div, pow, neg, exp, log, sin, cos, tanh, sqrt,
    square.

    Parameters
    ----------
    expr : tuple, str or number
        The expression DAG.
    values : mapping, optional
        Variable values; required if ``expr`` names any variable.

    Returns
    -------
    result : RichResult
        Keys: value, grad, node_grads, nodes, n_nodes, topo_order,
        fd_check, estimate, n, method.

    Examples
    --------
    ``f(x, y) = x*y + exp(x)`` at ``(0, 3)`` has value 1 and gradient
    ``(y + e^x, x) = (4, 0)``:

    >>> e = ("add", ("mul", "x", "y"), ("exp", "x"))
    >>> r = geron_computational_graph(e, {"x": 0.0, "y": 3.0})
    >>> r["value"]
    1.0
    >>> r["grad"]
    {'x': 4.0, 'y': 0.0}

    A shared sub-expression is stored once, so `x*x + x*x` costs 3 nodes,
    not 5:

    >>> sq = ("mul", "x", "x")
    >>> r2 = geron_computational_graph(("add", sq, sq), {"x": 3.0})
    >>> r2["value"], r2["grad"]["x"]
    (18.0, 12.0)
    >>> r2["n_nodes"]
    3

    The reverse sweep agrees with a central finite difference:

    >>> r3 = geron_computational_graph(("log", ("add", ("square", "x"), 1.0)), {"x": 2.0})
    >>> round(r3["value"], 9), round(r3["grad"]["x"], 9)
    (1.609437912, 0.8)
    >>> abs(r3["fd_check"]["x"] - r3["grad"]["x"]) < 1e-6
    True

    References
    ----------
    Géron Appendix A
    """
    env = dict(values or {})
    nodes = []
    root = _forward(expr, env, nodes, {})
    value = nodes[root]["value"]

    adj = [0.0] * len(nodes)
    adj[root] = 1.0
    for i in range(len(nodes) - 1, -1, -1):
        g = adj[i]
        nd = nodes[i]
        if g == 0.0 or not nd["inputs"]:
            continue
        op = nd["op"]
        ins = nd["inputs"]
        vals = [nodes[j]["value"] for j in ins]
        if op == "add":
            adj[ins[0]] += g
            adj[ins[1]] += g
        elif op == "sub":
            adj[ins[0]] += g
            adj[ins[1]] -= g
        elif op == "mul":
            adj[ins[0]] += g * vals[1]
            adj[ins[1]] += g * vals[0]
        elif op == "div":
            adj[ins[0]] += g / vals[1]
            adj[ins[1]] -= g * vals[0] / (vals[1] ** 2)
        elif op == "pow":
            a, b = vals
            adj[ins[0]] += g * b * a ** (b - 1)
            if a > 0:
                adj[ins[1]] += g * (a**b) * math.log(a)
        elif op == "neg":
            adj[ins[0]] -= g
        elif op == "exp":
            adj[ins[0]] += g * nd["value"]
        elif op == "log":
            adj[ins[0]] += g / vals[0]
        elif op == "sin":
            adj[ins[0]] += g * math.cos(vals[0])
        elif op == "cos":
            adj[ins[0]] -= g * math.sin(vals[0])
        elif op == "tanh":
            adj[ins[0]] += g * (1.0 - nd["value"] ** 2)
        elif op == "sqrt":
            adj[ins[0]] += g / (2.0 * nd["value"])
        elif op == "square":
            adj[ins[0]] += g * 2.0 * vals[0]

    grad = {}
    for i, nd in enumerate(nodes):
        if nd["op"] == "var":
            grad[nd["name"]] = grad.get(nd["name"], 0.0) + adj[i]

    fd = {}
    h = 1e-5
    for name in grad:
        up, dn = dict(env), dict(env)
        up[name] = env[name] + h
        dn[name] = env[name] - h
        nu, nv = [], []
        vu = nu[_forward(expr, up, nu, {})]["value"]
        vd = nv[_forward(expr, dn, nv, {})]["value"]
        fd[name] = (vu - vd) / (2 * h)

    return RichResult(
        title="Computational graph",
        summary_lines=[("Value", float(value)), ("Nodes", len(nodes))],
        interpretation="One reverse sweep gives every partial derivative, whatever the number of variables.",
        payload={
            "value": float(value),
            "grad": grad,
            "gradient": grad,
            "node_grads": adj,
            "nodes": nodes,
            "n_nodes": int(len(nodes)),
            "topo_order": list(range(len(nodes))),
            "fd_check": fd,
            "estimate": float(value),
            "n": int(len(nodes)),
            "method": "expression DAG with forward evaluation and reverse-mode adjoint sweep",
        },
    )


def cheatsheet():
    return "hmcgrf: Computational graph for expression differentiation"
