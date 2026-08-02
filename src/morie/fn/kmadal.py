# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Ch 4: AdaLoRA SVD-parametrized update with rank pruning."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_adalora_rank_allocation"]


def kamath_adalora_rank_allocation(P, s, Q, importance=None,
                                   target_rank=None):
    r"""Delta W = P diag(s) Q^T, with the least important s_i pruned.

    ``P`` is d x r, ``Q`` is k x r and ``s`` holds the r singular
    values. ``importance`` is the sensitivity score I(s_i) AdaLoRA
    budgets on (defaulting to |s_i|); the ``target_rank`` triplets
    with the highest importance are kept and the rest have their
    singular value set to 0 -- a real, exact rank reduction, checkable
    by ``numpy.linalg.matrix_rank``.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, AdaLoRA; Zhang et
    al. (2023).

    Examples
    --------
    >>> out = kamath_adalora_rank_allocation([[1.0, 0.0], [0.0, 1.0]],
    ...     [3.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], target_rank=1)
    >>> out["Delta_W"]
    [[3.0, 0.0], [0.0, 0.0]]
    >>> out["kept"]
    [0]
    """
    Pm = np.atleast_2d(np.asarray(P, dtype=float))
    Qm = np.atleast_2d(np.asarray(Q, dtype=float))
    sv = np.atleast_1d(np.asarray(s, dtype=float))
    r = sv.size
    if r == 0:
        raise ValueError("the rank is 0; there is no update to form.")
    if Pm.shape[1] != r or Qm.shape[1] != r:
        raise ValueError(
            f"P is {Pm.shape} and Q is {Qm.shape}, but s has {r} "
            "entries; both must have r columns.")
    imp = np.abs(sv) if importance is None else \
        np.atleast_1d(np.asarray(importance, dtype=float))
    if imp.size != r:
        raise ValueError(
            f"{imp.size} importance scores for {r} triplets.")
    if np.any(imp < 0):
        raise ValueError("importance scores cannot be negative.")
    tr = r if target_rank is None else int(target_rank)
    if not (0 <= tr <= r):
        raise ValueError(
            f"target_rank {tr} must lie in [0, r] with r = {r}.")
    keep = np.sort(np.argsort(-imp, kind="stable")[:tr])
    s_pruned = np.zeros_like(sv)
    s_pruned[keep] = sv[keep]
    dW = (Pm * s_pruned) @ Qm.T
    return RichResult(payload={
        "estimate": float(np.linalg.norm(dW)),
        "Delta_W": [[float(v) for v in row] for row in dW],
        "s_pruned": [float(v) for v in s_pruned],
        "kept": [int(i) for i in keep], "target_rank": tr,
        "effective_rank": int(np.count_nonzero(s_pruned)), "n": r,
        "method": "AdaLoRA SVD update with importance-based pruning "
                  "(Kamath Ch 4)"})


def cheatsheet():
    return "kmadal: P diag(s) Q^T after keeping the top-importance s_i"
