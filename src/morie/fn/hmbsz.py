# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batch size heuristic: power of two in [32, 512] balancing noise and throughput."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_batch_size_heuristic"]

_CANDIDATES = (32, 64, 128, 256, 512)


def geron_batch_size_heuristic(n_train, steps_per_epoch_target=10, memory_limit=None):
    """
    Batch size heuristic: power of two in [32, 512].

    Formula: B in {32, 64, 128, 256, 512}

    Picks the largest candidate B that still leaves at least
    `steps_per_epoch_target` gradient steps per epoch
    (B <= n_train / steps_per_epoch_target), clamped into the
    admissible set, and further capped by `memory_limit` if given.

    Parameters
    ----------
    n_train : int
        Number of training examples; must be >= 1.
    steps_per_epoch_target : int, default 10
        Minimum number of parameter updates wanted per epoch.
    memory_limit : int, optional
        Hard cap on batch size imposed by device memory.

    Returns
    -------
    result : RichResult
        Keys: batch_size, steps_per_epoch, candidates, estimate, n, method.

    Examples
    --------
    >>> r = geron_batch_size_heuristic(1000)
    >>> r["batch_size"]
    64
    >>> r["steps_per_epoch"]
    16
    >>> geron_batch_size_heuristic(50)["batch_size"]
    32
    >>> geron_batch_size_heuristic(10 ** 6)["batch_size"]
    512
    >>> geron_batch_size_heuristic(10 ** 6, memory_limit=100)["batch_size"]
    64

    References
    ----------
    Géron Ch 9
    """
    n = int(n_train)
    if n < 1:
        raise ValueError("geron_batch_size_heuristic: n_train must be >= 1")
    target = int(steps_per_epoch_target)
    if target < 1:
        raise ValueError("geron_batch_size_heuristic: steps_per_epoch_target must be >= 1")

    cap = n / target
    if memory_limit is not None:
        if int(memory_limit) < 1:
            raise ValueError("geron_batch_size_heuristic: memory_limit must be >= 1")
        cap = min(cap, float(int(memory_limit)))

    feasible = [b for b in _CANDIDATES if b <= cap]
    # Below the smallest candidate the heuristic floors at 32 (or at the
    # dataset size, when the dataset is smaller than one batch).
    batch = max(feasible) if feasible else min(32, n)
    steps = int(np.ceil(n / batch))

    return RichResult(
        title="Batch size heuristic",
        summary_lines=[("Batch size", batch), ("Steps per epoch", steps)],
        payload={
            "batch_size": batch,
            "steps_per_epoch": steps,
            "candidates": list(_CANDIDATES),
            "cap": float(cap),
            "clamped": not feasible,
            "estimate": float(batch),
            "n": n,
            "method": "Batch size heuristic (largest power of two in [32, 512] under the step-count cap)",
        },
    )


def cheatsheet():
    return "hmbsz: Batch size heuristic: power of two in [32, 512] balancing noise and throughput"
