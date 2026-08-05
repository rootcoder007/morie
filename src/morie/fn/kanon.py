# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""k-anonymity check.

Sweeney (2002), "k-anonymity: a model for protecting privacy",
International Journal of Uncertainty, Fuzziness and Knowledge-Based
Systems 10(5):557-570, doi:10.1142/S0218488502001648.  A release
satisfies k-anonymity when every combination of quasi-identifier
values appearing in it appears at least k times:

    min_g |g| >= k over the equivalence classes g induced by the
    quasi-identifiers.

The smallest class is therefore the whole diagnosis; the classes below
k are reported so the caller can see which records need generalising.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["k_anonymity_check"]


def k_anonymity_check(y, quasi_ids, k=5):
    """Smallest equivalence class size and whether it meets k.

    Parameters
    ----------
    y : array-like or None
        Sensitive attribute; not used by the criterion, kept for the
        module's interface.
    quasi_ids : n x q matrix
        Quasi-identifier values, one row per record.
    k : int
        Required minimum class size.
    """
    Q = core.mat(quasi_ids)
    n = len(Q)
    if n == 0:
        raise ValueError("k_anonymity_check: quasi_ids is empty")
    kk = int(k)
    if kk < 1:
        raise ValueError("k_anonymity_check: k must be at least 1")
    keys = []
    counts = []
    for i in range(n):
        key = tuple(Q[i])
        found = -1
        for j in range(len(keys)):
            if keys[j] == key:
                found = j
                break
        if found < 0:
            keys.append(key)
            counts.append(1)
        else:
            counts[found] += 1
    smallest = min(counts)
    violating = sum(1 for c in counts if c < kk)
    at_risk = sum(c for c in counts if c < kk)
    return RichResult(
        title="k-anonymity check",
        summary_lines=[("records", n), ("classes", len(counts)), ("k", kk)],
        payload={
            "estimate": smallest,
            "min_class_size": smallest,
            "class_sizes": counts,
            "n_classes": len(counts),
            "violating_classes": violating,
            "records_at_risk": at_risk,
            "satisfies_k": 1 if smallest >= kk else 0,
            "k": kk,
            "n": n,
            "method": "min class size over quasi-identifier equivalence classes, Sweeney (2002)",
        },
    )


def cheatsheet():
    return "kanon: k-anonymity check"
