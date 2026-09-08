# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Heterotrait-monotrait ratio of correlations (HTMT).

Henseler, Ringle and Sarstedt (2015), "A new criterion for assessing
discriminant validity in variance-based structural equation
modeling", Journal of the Academy of Marketing Science 43(1):115-135,
doi:10.1007/s11747-014-0403-8, equation (6): for constructs i and j
with K_i and K_j indicators,

    HTMT_ij = mean of the heterotrait-heteromethod correlations
              / sqrt( mean within-i correlation * mean within-j correlation ),

the averages running over distinct indicator pairs.  Discriminant
validity is questioned when the ratio exceeds 0.85 (or the more
lenient 0.90); the criterion is a decision, so the tests report a
confusion matrix rather than an average.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["htmt_ratio"]


def htmt_ratio(X, construct_assignment, threshold=0.85):
    """HTMT for every construct pair, and the discriminant validity verdict."""
    M = core.mat(X)
    n = len(M)
    if n < 3:
        raise ValueError("htmt_ratio: need at least three observations")
    p = len(M[0])
    g = [int(v) for v in core.vec(construct_assignment)]
    if len(g) != p:
        raise ValueError("htmt_ratio: one construct label per indicator is required")
    groups = sorted(set(g))
    if len(groups) < 2:
        raise ValueError("htmt_ratio: need at least two constructs")
    for lab in groups:
        if sum(1 for v in g if v == lab) < 2:
            raise ValueError("htmt_ratio: every construct needs at least two indicators")
    cols = [[M[i][j] for i in range(n)] for j in range(p)]
    R = [[core.corr(cols[i], cols[j]) for j in range(p)] for i in range(p)]
    pairs = []
    ratios = []
    flags = []
    for ai in range(len(groups)):
        for bi in range(ai + 1, len(groups)):
            A = [j for j in range(p) if g[j] == groups[ai]]
            B = [j for j in range(p) if g[j] == groups[bi]]
            hetero_v = [abs(R[x][y]) for x in A for y in B]
            wa = [abs(R[A[x]][A[y]]) for x in range(len(A)) for y in range(x + 1, len(A))]
            wb = [abs(R[B[x]][B[y]]) for x in range(len(B)) for y in range(x + 1, len(B))]
            mh = sum(hetero_v) / len(hetero_v)
            ma = sum(wa) / len(wa)
            mb = sum(wb) / len(wb)
            den = math.sqrt(ma * mb)
            v = float("inf") if den == 0 else mh / den
            pairs.append((groups[ai], groups[bi]))
            ratios.append(v)
            flags.append(1 if v > float(threshold) else 0)
    worst = max(ratios)
    return RichResult(
        title="Heterotrait-monotrait ratio",
        summary_lines=[("constructs", len(groups)), ("pairs", len(ratios))],
        payload={
            "estimate": worst,
            "htmt": ratios,
            "pair_first": [pairs[0][0], pairs[0][1]],
            "flagged": flags,
            "threshold": float(threshold),
            "discriminant_validity": 1 if worst <= float(threshold) else 0,
            "n": n,
            "method": "HTMT eq. (6) of Henseler, Ringle & Sarstedt (2015)",
        },
    )


def cheatsheet():
    return "hetero: heterotrait-monotrait ratio (HTMT)"


# compact alias per ledger/NAMING.md
htmtratio = htmt_ratio
