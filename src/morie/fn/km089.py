# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.13: Social Group Substitution invariance."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_sgs_invariance"]


def kamath_ch6_sgs_invariance(Yhat_i, Yhat_j, psi=None):
    """SGS = psi(Yhat_i, Yhat_j), the invariance of the output under a
    demographic substitution.

    ``psi`` defaults to exact match and must return a value in [0, 1]
    (or a bool) per pair. Sequences are compared elementwise and the
    mean invariance reported; 1 means the counterfactual changed
    nothing, which is the fairness criterion.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.13, printed
    p. 236.

    Examples
    --------
    >>> out = kamath_ch6_sgs_invariance(["a", "b"], ["a", "c"])
    >>> out["estimate"], out["per_pair"]
    (0.5, [1.0, 0.0])
    >>> kamath_ch6_sgs_invariance("she is a doctor", "he is a doctor",
    ...     psi=lambda u, v: float(u.split()[-1] == v.split()[-1])
    ...     )["estimate"]
    1.0
    """
    match = (lambda u, v: float(u == v)) if psi is None else psi
    if not callable(match):
        raise ValueError("psi must be a callable (Yhat_i, Yhat_j) -> "
                         "invariance in [0, 1].")
    if isinstance(Yhat_i, str) or isinstance(Yhat_j, str):
        pairs = [(Yhat_i, Yhat_j)]
    else:
        a, b = list(Yhat_i), list(Yhat_j)
        if not a:
            raise ValueError("no outputs to compare.")
        if len(a) != len(b):
            raise ValueError(
                f"the original produced {len(a)} outputs and the "
                f"counterfactual {len(b)}; they must be paired.")
        pairs = list(zip(a, b))
    vals = []
    for u, v in pairs:
        r = float(match(u, v))
        if not (0.0 <= r <= 1.0):
            raise ValueError(
                f"psi returned {r:.6g}; an invariance metric lies in "
                "[0, 1].")
        vals.append(r)
    arr = np.asarray(vals, dtype=float)
    return RichResult(payload={
        "estimate": float(arr.mean()), "per_pair": vals,
        "n_invariant": int(np.sum(arr == 1.0)), "n": len(vals),
        "method": "Social Group Substitution invariance (Kamath Eq 6.13)"})


def cheatsheet():
    return "km089: psi(original, counterfactual), 1 = unchanged"
