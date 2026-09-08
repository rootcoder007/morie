# morie.fn -- slice s03 (rootcoder007/morie)
"""Spearman-Brown prophecy formula.

Source consulted: Spearman, C. (1910).  Correlation calculated from
faulty data.  *British Journal of Psychology* 3, 271-295, and Brown, W.
(1910).  Some experimental results in the correlation of mental
abilities.  *British Journal of Psychology* 3, 296-322.  The two papers
give the same result independently:

    r' = k r / (1 + (k - 1) r)

for the reliability of a test lengthened by a factor k, where r is the
reliability of the original test.  Both 1910 papers are out of copyright
but were not available as a full text here; the equation is quoted in
its standard published form, which is unambiguous.

The inverse -- how much a test must be lengthened to reach a target
reliability -- follows by solving for k and is returned as ``k_needed``
when a target is supplied.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = ["spearman_brown"]


def spearman_brown(r, k, target=None):
    """Projected reliability of a test lengthened by a factor k.

    Parameters
    ----------
    r : float
        Reliability of the original test, in [0, 1].
    k : float
        Lengthening factor (k = 2 doubles the number of items).
    target : float, optional
        If given, also report the factor k needed to reach this
        reliability, k = target (1 - r) / (r (1 - target)).

    Returns
    -------
    RichResult with payload:
        estimate  : r' = k r / (1 + (k - 1) r)
        r, k
        k_needed  : lengthening factor for ``target`` (nan if not given)
    """
    r = float(r)
    k = float(k)
    den = 1.0 + (k - 1.0) * r
    est = (k * r) / den if den != 0.0 else float("nan")
    if target is None:
        kneed = float("nan")
    else:
        t = float(target)
        d2 = r * (1.0 - t)
        kneed = (t * (1.0 - r)) / d2 if d2 != 0.0 else float("nan")
    return RichResult(
        title="Spearman-Brown prophecy formula",
        summary_lines=[("projected reliability", est)],
        payload={
            "estimate": est,
            "r": r,
            "k": k,
            "k_needed": kneed,
            "method": "Spearman-Brown projected reliability",
        },
    )


def cheatsheet():
    return "sbreli: Spearman-Brown projected reliability"


spearmanbrown = spearman_brown
