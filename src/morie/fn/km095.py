# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.19: the gender direction in embedding space."""

from . import _array_core as np

from ._richresult import RichResult
from .km094 import _pair_vectors

__all__ = ["kamath_ch6_gender_direction"]


def kamath_ch6_gender_direction(A, E):
    """g = (1/|A|) sum_{(a_i,a_j) in A} (E(a_j) - E(a_i)).

    The average feminine -> masculine displacement, one vector standing
    in for the whole axis. ORDER MATTERS: a_i is the
    feminine-associated word and a_j the masculine one, so swapping the
    pair negates g. The pair handling is km094's, delegated. A norm of
    0 means the pairs cancelled and there is no direction to project
    onto -- reported, since km096 would divide by it.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.19, printed
    p. 243.

    Examples
    --------
    >>> E = {"she": [0.0, 1.0], "he": [2.0, 1.0]}
    >>> out = kamath_ch6_gender_direction([("she", "he")], E)
    >>> out["g"], out["norm"]
    ([2.0, 0.0], 2.0)
    """
    pairs = _pair_vectors(A, E, "E")
    diffs = np.asarray([vj - vi for vi, vj in pairs], dtype=float)
    g = diffs.mean(axis=0)
    norm = float(np.linalg.norm(g))
    return RichResult(payload={
        "g": [float(v) for v in g], "norm": norm,
        "per_pair": [[float(v) for v in d] for d in diffs],
        "degenerate": norm == 0.0,
        "estimate": norm, "n": len(pairs),
        "method": "gender direction (Kamath Eq 6.19)"})


def cheatsheet():
    return "km095: g = mean(E(masculine) - E(feminine)) over pairs"
