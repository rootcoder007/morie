# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.18: the counterfactual-pair debiasing regulariser."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_debias_regularizer"]


def _embed(E, name):
    """Resolve E into a callable word -> vector; km095/km096 reuse."""
    if callable(E):
        return E
    if isinstance(E, dict):
        def lookup(a):
            if a not in E:
                raise ValueError(f"{a!r} has no embedding in {name}.")
            return E[a]
        return lookup
    raise ValueError(f"{name} must be a callable word -> vector or a "
                     "mapping.")


def _pair_vectors(A, E, name):
    emb = _embed(E, name)
    pairs = list(A)
    if not pairs:
        raise ValueError("A is empty; a sum over no counterfactual pairs "
                         "is undefined, not 0.")
    out = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(
                "every element of A must be a counterfactual PAIR "
                f"(a_i, a_j); got one of length {len(pair)}.")
        vi = np.atleast_1d(np.asarray(emb(pair[0]), dtype=float))
        vj = np.atleast_1d(np.asarray(emb(pair[1]), dtype=float))
        if vi.shape != vj.shape:
            raise ValueError(
                f"the pair {pair!r} has embeddings of shapes {vi.shape} "
                f"and {vj.shape}.")
        out.append((vi, vj))
    dim = out[0][0].shape
    if any(v.shape != dim for p in out for v in p):
        raise ValueError("all embeddings must share one dimension.")
    return out


def kamath_ch6_debias_regularizer(A, E, lam):
    """R = lam sum_{(a_i,a_j) in A} ||E(a_i) - E(a_j)||^2.

    Pulls each protected attribute onto its counterfactual: the penalty
    is 0 exactly when every pair is embedded identically. Squared
    distances, so one badly separated pair dominates -- the per-pair
    terms are returned for that reason.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.18, printed
    p. 242.

    Examples
    --------
    >>> E = {"man": [1.0, 0.0], "woman": [0.0, 1.0]}
    >>> out = kamath_ch6_debias_regularizer([("man", "woman")], E, 2.0)
    >>> out["estimate"], out["per_pair"]
    (4.0, [2.0])
    """
    lam = float(lam)
    if lam < 0 or not np.isfinite(lam):
        raise ValueError("lam must be finite and non-negative.")
    pairs = _pair_vectors(A, E, "E")
    per = [float(np.sum((vi - vj) ** 2)) for vi, vj in pairs]
    return RichResult(payload={
        "estimate": float(lam * sum(per)), "per_pair": per,
        "unweighted": float(sum(per)), "lam": lam, "n": len(per),
        "method": "counterfactual-pair debiasing regulariser "
                  "(Kamath Eq 6.18)"})


def cheatsheet():
    return "km094: lam * sum of squared distances within pairs"
