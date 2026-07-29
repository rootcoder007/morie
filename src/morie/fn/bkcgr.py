# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov Ch 1: forward and backward pass over a computational graph."""

import math

from ._richresult import RichResult

__all__ = ["burkov_computational_graph"]

_OPS = {
    "add": (lambda a, b: a + b, lambda a, b, g: (g, g)),
    "sub": (lambda a, b: a - b, lambda a, b, g: (g, -g)),
    "mul": (lambda a, b: a * b, lambda a, b, g: (g * b, g * a)),
    "tanh": (lambda a: math.tanh(a),
             lambda a, g: (g * (1.0 - math.tanh(a) ** 2),)),
    "sigmoid": (lambda a: 1.0 / (1.0 + math.exp(-a)),
                lambda a, g: (g * (1.0 / (1.0 + math.exp(-a)))
                              * (1.0 - 1.0 / (1.0 + math.exp(-a))),)),
    "relu": (lambda a: max(a, 0.0), lambda a, g: (g if a > 0 else 0.0,)),
    "square": (lambda a: a * a, lambda a, g: (2.0 * a * g,)),
    "log": (lambda a: math.log(a), lambda a, g: (g / a,)),
    "exp": (lambda a: math.exp(a), lambda a, g: (g * math.exp(a),)),
}


def burkov_computational_graph(graph, inputs):
    """One forward pass and one reverse-mode backward pass.

    ``graph`` is a topologically ordered list of nodes
    ``{"name": str, "op": str, "args": [names]}``; ``inputs`` maps leaf
    names to values. The last node is the output. Gradients of the
    output with respect to every leaf come back exactly (closed-form
    local derivatives, chain rule across the DAG); the tests pin them
    against central finite differences.

    References: Burkov LM (2025), Ch 1, computational graph.

    Examples
    --------
    >>> g = [{"name": "s", "op": "mul", "args": ["x", "y"]},
    ...      {"name": "out", "op": "tanh", "args": ["s"]}]
    >>> out = burkov_computational_graph(g, {"x": 0.5, "y": 0.0})
    >>> out["gradients"]["y"]
    0.5
    """
    if not graph:
        raise ValueError("the graph is empty.")
    values = dict(inputs)
    for node in graph:
        op = node["op"]
        if op not in _OPS:
            raise ValueError(
                f"unknown op {op!r}; supported: {sorted(_OPS)}.")
        args = node["args"]
        fwd = _OPS[op][0]
        missing = [a for a in args if a not in values]
        if missing:
            raise ValueError(
                f"node {node['name']!r} needs {missing} before they are "
                "computed; the graph must be topologically ordered.")
        values[node["name"]] = fwd(*(values[a] for a in args))
    out_name = graph[-1]["name"]
    grads = {name: 0.0 for name in values}
    grads[out_name] = 1.0
    for node in reversed(graph):
        op = node["op"]; args = node["args"]
        bwd = _OPS[op][1]
        g = grads[node["name"]]
        local = bwd(*[values[a] for a in args], g)
        for a, ga in zip(args, local):
            grads[a] += ga
    leaf_grads = {k: grads[k] for k in inputs}
    return RichResult(payload={
        "output": values[out_name], "estimate": values[out_name],
        "gradients": leaf_grads, "values": values, "n": len(graph),
        "method": "Computational-graph autodiff (Burkov Ch 1)"})


def cheatsheet():
    return "bkcgr: forward + reverse-mode pass over a computation DAG (Burkov Ch 1)"
