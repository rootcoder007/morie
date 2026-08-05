# morie.fn -- wave 2 slice f_00 (rootcoder007/morie)
"""Advanced composition theorem for differential privacy.

Source: Dwork, C., Rothblum, G. N. and Vadhan, S. (2010), "Boosting and
differential privacy", 51st IEEE Symposium on Foundations of Computer
Science (FOCS), 51-60, doi:10.1109/FOCS.2010.12.  The full text was not
retrievable here, so the theorem is written in the standard published
form the module specification states: the k-fold adaptive composition of
mechanisms each (epsilon, delta)-differentially private is
(epsilon', k delta + delta')-differentially private with

    epsilon' = sqrt( 2 k ln(1 / delta') ) epsilon + k epsilon (e^epsilon - 1).

Two facts make the result worth having, and both are checked as anchors.

The leading term scales as sqrt(k), not k.  Basic (sequential)
composition gives k epsilon; advanced composition trades a small
delta' for a privacy loss that grows like the square root of the number
of queries.  As epsilon goes to zero the second term is order
k epsilon^2 and vanishes relative to the first, so the ratio of
epsilon' to epsilon sqrt(2 k ln(1/delta')) tends to 1 -- that limit is
the anchor for the leading term.

Advanced composition is not uniformly better.  For small k or large
epsilon the quadratic correction dominates and k epsilon is the tighter
bound; ``epsilon_basic`` and ``tighter`` are returned so the crossover is
visible rather than assumed, because quoting the advanced bound where
the basic one is smaller overstates the privacy loss.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core  # noqa: F401

from ._richresult import RichResult

__all__ = ["advanced_composition"]


def advanced_composition(epsilon, delta=0.0, k=1, delta_prime=1e-6):
    """Privacy parameters after k-fold adaptive composition.

    Parameters
    ----------
    epsilon : float
        Per-mechanism epsilon, positive.
    delta : float
        Per-mechanism delta, in [0, 1).
    k : int
        Number of compositions, at least one.
    delta_prime : float
        Slack traded for the sqrt(k) rate, in (0, 1].

    Returns
    -------
    epsilon_total : the advanced-composition epsilon'
    delta_total : k delta + delta'
    epsilon_basic : k epsilon, the sequential-composition bound
    tighter : "advanced" or "basic", whichever bound is smaller
    """
    e = float(epsilon)
    if not (e > 0.0):
        raise ValueError("advanced_composition: epsilon must be positive")
    d = float(delta)
    if not (0.0 <= d < 1.0):
        raise ValueError("advanced_composition: delta must lie in [0, 1)")
    kk = int(k)
    if kk < 1:
        raise ValueError("advanced_composition: k must be at least one")
    dp = float(delta_prime)
    if not (0.0 < dp <= 1.0):
        raise ValueError("advanced_composition: delta_prime must lie in (0, 1]")
    lead = math.sqrt(2.0 * kk * math.log(1.0 / dp)) * e
    quad = kk * e * (math.exp(e) - 1.0)
    et = lead + quad
    eb = kk * e
    dt = kk * d + dp
    return RichResult(
        title="Advanced composition",
        summary_lines=[("epsilon_total", et), ("delta_total", dt)],
        payload={
            "epsilon_total": et,
            "estimate": et,
            "delta_total": dt,
            "epsilon_basic": eb,
            "delta_basic": kk * d,
            "leading_term": lead,
            "quadratic_term": quad,
            "tighter": "advanced" if et < eb else "basic",
            "epsilon_effective": et if et < eb else eb,
            "k": kk,
            "method": "Dwork, Rothblum and Vadhan (2010) advanced composition",
        },
    )


def cheatsheet():
    return "advcmp: Advanced composition theorem for differential privacy"


# compact alias per ledger/NAMING.md
advancedcomposition = advanced_composition
