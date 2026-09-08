# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Horvitz-Thompson variance of the estimated total.

Horvitz and Thompson (1952), "A generalization of sampling without
replacement from a finite universe", JASA 47(260):663-685,
doi:10.1080/01621459.1952.10483446, equation (4.2) in the
Sen-Yates-Grundy-free ("HT") form

    V = sum_i sum_j (pi_ij - pi_i pi_j) (y_i / pi_i) (y_j / pi_j) / pi_ij,

with pi_ii = pi_i, so the diagonal contributes
(1 - pi_i) y_i^2 / pi_i^2.  For simple random sampling without
replacement the whole expression collapses to N^2 (1 - n/N) S^2 / n,
which is the closed form the tests check against.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["ht_variance"]


def ht_variance(y, pi, pi_ij):
    """Horvitz-Thompson variance estimate of the total."""
    v = core.vec(y)
    p = core.vec(pi)
    P = core.mat(pi_ij)
    n = len(v)
    if n == 0:
        raise ValueError("ht_variance: y is empty")
    if len(p) != n:
        raise ValueError("ht_variance: y and pi have different lengths")
    if len(P) != n:
        raise ValueError("ht_variance: pi_ij must be n x n")
    for r in P:
        if len(r) != n:
            raise ValueError("ht_variance: pi_ij must be n x n")
    for x in p:
        if x <= 0 or x > 1:
            raise ValueError("ht_variance: inclusion probabilities must lie in (0, 1]")
    tot = 0.0
    for i in range(n):
        tot += v[i] / p[i]
    var = 0.0
    for i in range(n):
        for j in range(n):
            pij = p[i] if i == j else P[i][j]
            if pij <= 0:
                raise ValueError("ht_variance: joint inclusion probability is not positive")
            var += (pij - p[i] * p[j]) * (v[i] / p[i]) * (v[j] / p[j]) / pij
    se = var ** 0.5 if var >= 0 else float("nan")
    return RichResult(
        title="Horvitz-Thompson variance",
        summary_lines=[("n", n), ("total", tot)],
        payload={
            "estimate": var,
            "variance": var,
            "total": tot,
            "se": se,
            "n": n,
            "method": "sum_ij (pi_ij - pi_i pi_j)(y_i/pi_i)(y_j/pi_j)/pi_ij, Horvitz & Thompson (1952)",
        },
    )


def cheatsheet():
    return "htvar1: Horvitz-Thompson variance"


# compact alias per ledger/NAMING.md
htvariance = ht_variance
