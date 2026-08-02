# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""TorchScript: statically-typed graph representation of PyTorch models."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_torchscript", "trace", "run_graph"]

_KINDS = ("linear", "bias", "relu", "tanh", "sigmoid")


def _apply(op, x):
    kind = op[0] if isinstance(op, (tuple, list)) else "callable"
    if kind == "linear":
        return x @ np.asarray(op[1], dtype=float)
    if kind == "bias":
        return x + np.asarray(op[1], dtype=float)
    if kind == "relu":
        return np.maximum(x, 0.0)
    if kind == "tanh":
        return np.tanh(x)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))
    return op(x)


def trace(model, example_inputs):
    """Record a typed graph by running `model`'s ops on `example_inputs`.

    This is what tracing *is*: the ops are executed once and the observed
    input/output shapes are frozen into the graph. Control flow that
    depends on the data is therefore invisible to the trace -- the known
    limitation of ``torch.jit.trace`` -- so the graph is only valid for
    inputs of the recorded shape.
    """
    x = np.asarray(example_inputs, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.size == 0:
        raise ValueError("trace: example_inputs is empty; tracing needs a real tensor")
    graph = []
    for i, op in enumerate(model):
        if isinstance(op, (tuple, list)):
            if op[0] not in _KINDS:
                raise ValueError(f"trace: op {i} has unknown kind {op[0]!r}; known: {', '.join(_KINDS)}")
            if op[0] in ("linear", "bias") and len(op) < 2:
                raise ValueError(f"trace: op {i} of kind {op[0]!r} needs a parameter tensor")
        elif not callable(op):
            raise ValueError(f"trace: op {i} must be a (kind, param) tuple or a callable")
        in_shape = x.shape
        y = np.asarray(_apply(op, x), dtype=float)
        if not np.all(np.isfinite(y)):
            raise ValueError(f"trace: op {i} produced non-finite values")
        graph.append(
            {
                "index": i,
                "kind": op[0] if isinstance(op, (tuple, list)) else "callable",
                "in_shape": tuple(in_shape),
                "out_shape": tuple(y.shape),
                "op": op,
            }
        )
        x = y
    return graph, x


def run_graph(graph, x):
    """Re-execute a traced graph, enforcing the recorded shapes."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    for node in graph:
        if a.shape[1:] != node["in_shape"][1:]:
            raise ValueError(
                f"run_graph: node {node['index']} was traced with input shape {node['in_shape']} but got {a.shape}; "
                "a traced graph is only valid for the shapes it was recorded with"
            )
        a = np.asarray(_apply(node["op"], a), dtype=float)
    return a


def geron_torchscript(model, example_inputs):
    """
    TorchScript: statically-typed graph representation of PyTorch models.

    Formula: torch.jit.trace(model) or torch.jit.script(model)

    A numpy-native stand-in for tracing (torch is not a dependency of
    morie, so the *mechanism* is implemented rather than the API called).
    `model` is a sequence of ops -- ``("linear", W)``, ``("bias", b)``,
    ``("relu",)``, ``("tanh",)``, ``("sigmoid",)`` or a plain callable --
    and tracing runs them once on `example_inputs`, freezing each node's
    input and output shape into a static graph.

    The round trip is then verified: the graph is re-executed and its
    output compared with the eager run, and re-running it at a different
    feature width raises instead of silently broadcasting. That is the
    real property of a traced graph -- it is specialised to the shapes it
    saw, and data-dependent control flow is baked out.

    Parameters
    ----------
    model : sequence
        Ops as described above (non-empty).
    example_inputs : array-like
        Representative input (n, d).

    Returns
    -------
    result : RichResult
        Keys: graph, output, replay, max_diff, n_nodes, shapes,
        estimate, n, method.

    Examples
    --------
    >>> W = [[1.0, 0.0], [0.0, 1.0]]
    >>> r = geron_torchscript([("linear", W), ("relu",)], [[1.0, -1.0]])
    >>> int(r["n_nodes"])
    2
    >>> r["output"].tolist()
    [[1.0, 0.0]]
    >>> round(float(r["max_diff"]), 12)
    0.0
    >>> r["shapes"]
    [((1, 2), (1, 2)), ((1, 2), (1, 2))]

    The graph refuses an input whose width it never saw:

    >>> run_graph(r["graph"], [[1.0, 2.0, 3.0]])
    Traceback (most recent call last):
        ...
    ValueError: run_graph: node 0 was traced with input shape (1, 2) but got (1, 3); a traced graph is only valid for the shapes it was recorded with

    References
    ----------
    Géron Appendix B
    """
    ops = list(model)
    if not ops:
        raise ValueError("geron_torchscript: model has no ops to trace")
    graph, out = trace(ops, example_inputs)
    replay = run_graph(graph, example_inputs)
    diff = float(np.max(np.abs(replay - out)))

    return RichResult(
        title="Traced graph (TorchScript-style)",
        summary_lines=[
            ("Nodes", len(graph)),
            ("Input shape", graph[0]["in_shape"]),
            ("Output shape", graph[-1]["out_shape"]),
            ("Replay max difference", diff),
        ],
        interpretation=(
            "Tracing records one execution: shapes are frozen and data-dependent branches disappear, "
            "which is why torch.jit.script exists for models whose control flow depends on the data."
        ),
        payload={
            "graph": graph,
            "output": out,
            "replay": replay,
            "max_diff": diff,
            "n_nodes": len(graph),
            "shapes": [(g["in_shape"], g["out_shape"]) for g in graph],
            "estimate": diff,
            "n": int(np.asarray(example_inputs, dtype=float).reshape(1, -1).shape[0]) if np.asarray(example_inputs).ndim == 1 else int(np.asarray(example_inputs).shape[0]),
            "method": "Shape-specialised tracing of an op sequence into a static graph, verified by replay",
        },
    )


def cheatsheet():
    return "hmtsc: TorchScript: statically-typed graph representation of PyTorch models"
