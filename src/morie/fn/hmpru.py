# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Magnitude-based weight pruning."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_weight_pruning"]


def geron_weight_pruning(model, sparsity, n_rounds=1):
    """
    Magnitude-based weight pruning.

    Formula: zero out |w| < threshold; optional iterative pruning + fine-tuning

    The threshold is GLOBAL across every tensor, not per layer: layers
    differ by orders of magnitude in weight scale, and a per-layer
    quantile prunes the same fraction everywhere whether or not that
    layer has anything to spare. Exactly ``sparsity`` of the weights are
    removed even when magnitudes tie, because the cut is made on a rank,
    not on a comparison.

    One-shot pruning to a high sparsity destroys accuracy; the schedule
    in ``schedule`` is the cubic ramp used to reach the target over
    ``n_rounds`` prune/fine-tune cycles. Fine-tuning itself is the
    caller's job -- this function does not train.

    Parameters
    ----------
    model : array-like, mapping of arrays, or sequence of arrays
        Weights to prune.
    sparsity : float
        Target fraction of zeros in [0, 1).
    n_rounds : int, default 1
        Rounds in the returned pruning schedule.

    Returns
    -------
    result : RichResult
        Keys: pruned, mask, threshold, achieved_sparsity, n_pruned,
        schedule, estimate, n, method.

    Examples
    --------
    Half of four weights: the two smallest magnitudes go, and the cut
    sits at |w| = 2.

    >>> r = geron_weight_pruning([1.0, -2.0, 3.0, -4.0], 0.5)
    >>> [float(v) for v in r["pruned"]]
    [0.0, 0.0, 3.0, -4.0]
    >>> float(r["threshold"]), float(r["achieved_sparsity"]), int(r["n_pruned"])
    (2.0, 0.5, 2)

    A dict of tensors is pruned against one global threshold:

    >>> d = geron_weight_pruning({"a": [0.1, 0.2], "b": [5.0, 6.0]}, 0.5)
    >>> [float(v) for v in d["pruned"]["a"]], [float(v) for v in d["pruned"]["b"]]
    ([0.0, 0.0], [5.0, 6.0])

    References
    ----------
    Geron Ch 17
    """
    sp = float(sparsity)
    if not (0.0 <= sp < 1.0):
        raise ValueError(f"geron_weight_pruning: sparsity must lie in [0, 1), got {sparsity!r}")
    R = int(n_rounds)
    if R < 1:
        raise ValueError(f"geron_weight_pruning: n_rounds must be >= 1, got {n_rounds!r}")

    if hasattr(model, "items"):
        keys = list(model.keys())
        tensors = [np.atleast_1d(np.asarray(model[k], dtype=float)) for k in keys]
        kind = "dict"
    elif isinstance(model, (list, tuple)) and model and isinstance(model[0], (list, tuple, np.ndarray)):
        keys = None
        tensors = [np.atleast_1d(np.asarray(t, dtype=float)) for t in model]
        kind = "list"
    else:
        keys = None
        tensors = [np.atleast_1d(np.asarray(model, dtype=float))]
        kind = "array"
    if sum(t.size for t in tensors) == 0:
        raise ValueError("geron_weight_pruning: model has no weights")
    for t in tensors:
        if not np.all(np.isfinite(t)):
            raise ValueError("geron_weight_pruning: model contains non-finite weights")

    flat = np.concatenate([np.abs(t).ravel() for t in tensors])
    N = flat.size
    k = int(np.floor(sp * N))
    order = np.argsort(flat, kind="mergesort")
    cut = np.zeros(N, dtype=bool)
    cut[order[:k]] = True
    thr = float(flat[order[k - 1]]) if k > 0 else 0.0

    out = []
    masks = []
    pos = 0
    for t in tensors:
        m = ~cut[pos : pos + t.size].reshape(t.shape)
        masks.append(m)
        out.append(np.where(m, t, 0.0))
        pos += t.size

    if kind == "dict":
        pruned = {kk: v for kk, v in zip(keys, out)}
        mask = {kk: v for kk, v in zip(keys, masks)}
    elif kind == "list":
        pruned, mask = out, masks
    else:
        pruned, mask = out[0], masks[0]

    sched = [float(sp * (1.0 - (1.0 - (i + 1) / R) ** 3)) for i in range(R)]
    return RichResult(
        title="Magnitude weight pruning",
        summary_lines=[("Weights", int(N)), ("Pruned", int(k)), ("Achieved sparsity", k / N)],
        interpretation="One global threshold, and a cubic ramp when the target is high enough to need fine-tuning.",
        payload={
            "pruned": pruned,
            "mask": mask,
            "threshold": thr,
            "achieved_sparsity": k / N,
            "n_pruned": int(k),
            "n_weights": int(N),
            "schedule": sched,
            "estimate": k / N,
            "n": int(N),
            "method": "Global magnitude pruning to an exact sparsity, with a cubic ramp schedule",
        },
    )


def cheatsheet():
    return "hmpru: Magnitude-based weight pruning to a target sparsity"
