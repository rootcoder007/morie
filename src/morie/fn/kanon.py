# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""k-anonymity check on a set of quasi-identifiers.

Sweeney (2002), "k-anonymity: a model for protecting privacy",
International Journal of Uncertainty, Fuzziness and Knowledge-Based
Systems 10(5):557-570, doi:10.1142/S0218488502001648.  A release
satisfies k-anonymity when every combination of quasi-identifier values
that appears in it appears at least k times:

    min over equivalence classes of |class| >= k.

The equivalence classes are the distinct rows of the quasi-identifier
block; the smallest of them is the binding constraint, and the rows in
classes below k are exactly the records that would have to be
suppressed or generalised.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["k_anonymity_check"]


def k_anonymity_check(y, quasi_ids, k):
    """Smallest equivalence class size and whether it reaches k.

    Parameters
    ----------
    y : array-like
        Records; only its length is used, to check against the block.
    quasi_ids : array-like
        Quasi-identifier block, one row per record.
    k : int
        Required anonymity level.
    """
    yv = core.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("k_anonymity_check: y is empty")
    rows = core.mat(quasi_ids)
    if len(rows) != n:
        raise ValueError("k_anonymity_check: quasi_ids and y have different lengths")
    kk = int(k)
    if kk < 1:
        raise ValueError("k_anonymity_check: k must be at least 1")
    counts = {}
    order = []
    for row in rows:
        key = "|".join("%.12g" % v for v in row)
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += 1
    sizes = [counts[key] for key in order]
    mn = min(sizes)
    viol = sum(s for s in sizes if s < kk)
    return RichResult(
        title="k-anonymity check",
        summary_lines=[("classes", len(sizes)), ("min class size", mn), ("k", kk)],
        payload={
            "estimate": float(mn),
            "min_class_size": float(mn),
            "max_class_size": float(max(sizes)),
            "mean_class_size": sum(sizes) / float(len(sizes)),
            "k": float(kk),
            "satisfies": 1.0 if mn >= kk else 0.0,
            "n_classes": float(len(sizes)),
            "n_violating": float(viol),
            "n_unique": float(sum(1 for s in sizes if s == 1)),
            "n": n,
            "method": "min over equivalence classes of |class| >= k, Sweeney (2002)",
        },
    )


def cheatsheet():
    return "kanon: k-anonymity check on a quasi-identifier set"


# compact alias per ledger/NAMING.md
kanonymity = k_anonymity_check
