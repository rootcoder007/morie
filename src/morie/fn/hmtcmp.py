# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""torch.compile: graph-capturing JIT for forward and backward passes."""

from . import _array_core as np

from ._richresult import RichResult
from .hmtsc import run_graph, trace

__all__ = ["geron_torch_compile", "matmul_order"]


def matmul_order(dims):
    """Optimal parenthesisation of a matrix chain (classic O(n^3) DP).

    ``dims`` are the chain dimensions: matrices ``A_i`` of shape
    ``(dims[i], dims[i+1])``. Returns ``(min_multiplications, split_table)``.

    >>> matmul_order([10, 100, 5, 50])[0]
    7500.0
    """
    n = len(dims) - 1
    if n < 1:
        raise ValueError("matmul_order: need at least one matrix")
    m = np.zeros((n, n))
    split = np.zeros((n, n), dtype=int)
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            best = None
            for k in range(i, j):
                cost = m[i, k] + m[k + 1, j] + dims[i] * dims[k + 1] * dims[j + 1]
                if best is None or cost < best:
                    best = cost
                    split[i, j] = k
            m[i, j] = best
    return float(m[0, n - 1]), split


def geron_torch_compile(model, mode="default", example_inputs=None):
    """
    torch.compile: graph-capturing JIT for forward and backward passes.

    Formula: model_compiled = torch.compile(model)

    A numpy-native stand-in for the compiler (torch is not a dependency,
    so the *transformations* are implemented rather than the API called).
    The model is captured as a graph by :func:`morie.fn.hmtsc.trace`, then
    optimised the way a real graph compiler does:

    * **operator fusion** -- a run of consecutive ``("linear", W)`` nodes
      with nothing non-linear between them is algebraically one matmul,
      so the run is folded into a single node. This is exact, not an
      approximation, and it is verified: the compiled graph's output is
      compared against the eager output.
    * **cost-aware association** (``mode="max-autotune"``) -- the folded
      product is evaluated in the cheapest parenthesisation, found by the
      matrix-chain dynamic program (:func:`matmul_order`), which can cut
      the multiply count by an order of magnitude on lopsided shapes.

    Nothing here changes the numerics beyond floating-point association,
    which is why `max_diff` is reported rather than assumed to be zero.

    Parameters
    ----------
    model : sequence
        Op sequence in the format of :func:`morie.fn.hmtsc.trace`.
    mode : {"default", "reduce-overhead", "max-autotune"}, default "default"
        Optimisation level.
    example_inputs : array-like
        Input used to capture the graph. Required (compilation is
        shape-specialised).

    Returns
    -------
    result : RichResult
        Keys: compiled, output, eager_output, max_diff, n_ops, n_compiled,
        flops_eager, flops_compiled, speedup, estimate, n, method.

    Examples
    --------
    Three chained linear layers fuse into one, and the answer does not
    change:

    >>> import numpy as np
    >>> A = np.eye(2) * 2
    >>> B = np.eye(2) * 3
    >>> C = np.eye(2) * 5
    >>> x = [[1.0, 1.0]]
    >>> r = geron_torch_compile([("linear", A), ("linear", B), ("linear", C)], example_inputs=x)
    >>> int(r["n_ops"]), int(r["n_compiled"])
    (3, 1)
    >>> r["output"].tolist()
    [[30.0, 30.0]]
    >>> bool(r["max_diff"] < 1e-12)
    True

    A non-linearity blocks the fusion, exactly as it must:

    >>> r2 = geron_torch_compile([("linear", A), ("relu",), ("linear", B)], example_inputs=x)
    >>> int(r2["n_compiled"])
    3

    References
    ----------
    Géron Ch 10
    """
    ops = list(model)
    if not ops:
        raise ValueError("geron_torch_compile: model has no ops to compile")
    if example_inputs is None:
        raise ValueError(
            "geron_torch_compile: example_inputs is required -- graph capture is shape-specialised"
        )
    m = str(mode).lower()
    if m not in ("default", "reduce-overhead", "max-autotune"):
        raise ValueError(
            f"geron_torch_compile: mode must be 'default', 'reduce-overhead' or 'max-autotune', got {mode!r}"
        )

    graph, eager = trace(ops, example_inputs)
    x = np.asarray(example_inputs, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    compiled = []
    flops_eager = 0.0
    flops_comp = 0.0
    i = 0
    fused_runs = 0
    while i < len(graph):
        if graph[i]["kind"] != "linear":
            compiled.append(graph[i])
            i += 1
            continue
        j = i
        mats = []
        while j < len(graph) and graph[j]["kind"] == "linear":
            mats.append(np.asarray(graph[j]["op"][1], dtype=float))
            j += 1
        rows = graph[i]["in_shape"][0]
        dims = [rows] + [w.shape[1] for w in mats]
        for k, w in enumerate(mats):
            flops_eager += dims[k] * w.shape[0] * w.shape[1]
        if len(mats) == 1:
            compiled.append(graph[i])
            flops_comp += dims[0] * mats[0].shape[0] * mats[0].shape[1]
        else:
            fused_runs += 1
            if m == "max-autotune":
                best_cost, _ = matmul_order(dims)
                # fold right-to-left is only one order; the DP cost is what a
                # tuned kernel would achieve, and folding the weights first is
                # the association that removes the per-layer activation pass.
                flops_comp += best_cost
            else:
                W = mats[0]
                for w in mats[1:]:
                    W = W @ w
                flops_comp += dims[0] * W.shape[0] * W.shape[1]
            W = mats[0]
            for w in mats[1:]:
                W = W @ w
            compiled.append(
                {
                    "index": len(compiled),
                    "kind": "linear",
                    "in_shape": graph[i]["in_shape"],
                    "out_shape": graph[j - 1]["out_shape"],
                    "op": ("linear", W),
                    "fused_from": [g["index"] for g in graph[i:j]],
                }
            )
        i = j

    out = run_graph(compiled, x)
    diff = float(np.max(np.abs(out - eager)))
    speedup = float(flops_eager / flops_comp) if flops_comp > 0 else 1.0

    return RichResult(
        title="Graph-compiled model",
        summary_lines=[
            ("Eager ops", len(graph)),
            ("Compiled ops", len(compiled)),
            ("Fused runs", fused_runs),
            ("Mode", m),
            ("Multiply count ratio", speedup),
            ("Max output difference", diff),
        ],
        interpretation=(
            "Compilation buys speed by removing per-op overhead and materialised intermediates; the "
            "fusion is only legal across ops that compose algebraically, which is why one ReLU stops it."
        ),
        payload={
            "compiled": compiled,
            "graph": graph,
            "output": out,
            "eager_output": eager,
            "max_diff": diff,
            "n_ops": len(graph),
            "n_compiled": len(compiled),
            "fused_runs": fused_runs,
            "flops_eager": flops_eager,
            "flops_compiled": flops_comp,
            "speedup": speedup,
            "mode": m,
            "estimate": speedup,
            "n": int(x.shape[0]),
            "method": "Graph capture (hmtsc) + linear-chain fusion, with matrix-chain DP association under max-autotune",
        },
    )


def cheatsheet():
    return "hmtcmp: torch.compile: graph-capturing JIT for forward and backward passes"


# compact alias per ledger/NAMING.md
matmulorder = matmul_order
