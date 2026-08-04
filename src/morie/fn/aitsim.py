# morie.fn -- slice s03 (rootcoder007/morie)
"""Gini-Simpson diversity of a composition.

Source consulted: Simpson, E. H. (1949).  Measurement of diversity.
*Nature* 163, 688.  Simpson's one-page note defines the concentration

    lambda = sum_i p_i^2

as the probability that two individuals drawn at random belong to the
same class.  The complement 1 - lambda is the Gini-Simpson index, the
probability that they differ.  The 1949 note is paywalled; the two
expressions are quoted in their standard published form, which is not
in dispute.

The input is treated as a composition: if it does not already sum to
one it is closed (divided by its total), which is the operation
Aitchison calls the closure of a compositional vector.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["compositional_simpson"]


def compositional_simpson(x):
    """Gini-Simpson index D = 1 - sum p_i^2 of a composition or count vector.

    Parameters
    ----------
    x : array-like
        Proportions or counts by class.  Negative entries are dropped.

    Returns
    -------
    RichResult with payload:
        D           : 1 - sum p_i^2
        estimate    : same as D
        lambda_     : Simpson concentration sum p_i^2
        inv_simpson : 1 / lambda, the effective number of classes
        p           : the closed composition
    """
    raw = [v for v in k.vec(x) if v >= 0.0]
    tot = 0.0
    for v in raw:
        tot += v
    if tot <= 0.0:
        return RichResult(
            title="Gini-Simpson diversity",
            payload={
                "D": float("nan"),
                "estimate": float("nan"),
                "lambda_": float("nan"),
                "inv_simpson": float("nan"),
                "p": [],
                "n": 0,
                "method": "Gini-Simpson diversity index",
            },
        )
    p = [v / tot for v in raw]
    lam = 0.0
    for v in p:
        lam += v * v
    d = 1.0 - lam
    return RichResult(
        title="Gini-Simpson diversity",
        summary_lines=[("D", d), ("Simpson lambda", lam)],
        payload={
            "D": d,
            "estimate": d,
            "lambda_": lam,
            "inv_simpson": 1.0 / lam if lam > 0.0 else float("inf"),
            "p": p,
            "n": len(p),
            "method": "Gini-Simpson diversity index",
        },
    )


def cheatsheet():
    return "aitsim: Gini-Simpson diversity index"
