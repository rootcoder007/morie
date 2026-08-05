# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Composition of differentially private mechanisms.

Dwork and Roth (2014), Foundations and Trends in Theoretical Computer
Science 9(3-4):211-407, doi:10.1561/0400000042.  Basic composition
(Theorem 3.14) adds the budgets,

    eps_total = sum_i eps_i,

while advanced composition (Theorem 3.20) shows that k applications of
an (eps, 0)-mechanism are (eps', k delta')-DP with

    eps' = sqrt(2 k ln(1 / delta')) eps + k eps (exp(eps) - 1).

The advanced bound is the smaller of the two once k is large and eps
small -- that crossover is the whole reason the theorem is worth
having, and the tests check it rather than assume it.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["k_step_dp_composition"]


def k_step_dp_composition(y, epsilons, delta_prime=1e-6):
    """Basic and advanced composition of a sequence of privacy budgets.

    Parameters
    ----------
    y : array-like or None
        Unused payload placeholder kept for the module's interface.
    epsilons : array-like
        Per-mechanism epsilon values, all positive.
    delta_prime : float
        The extra delta spent by the advanced composition bound.
    """
    eps = core.vec(epsilons)
    k = len(eps)
    if k == 0:
        raise ValueError("k_step_dp_composition: epsilons is empty")
    for v in eps:
        if v <= 0:
            raise ValueError("k_step_dp_composition: every epsilon must be positive")
    dp = float(delta_prime)
    if not 0 < dp < 1:
        raise ValueError("k_step_dp_composition: delta_prime must lie in (0, 1)")
    basic = 0.0
    for v in eps:
        basic += v
    e = max(eps)
    advanced = math.sqrt(2.0 * k * math.log(1.0 / dp)) * e + k * e * (math.exp(e) - 1.0)
    return RichResult(
        title="Differential privacy composition",
        summary_lines=[("mechanisms", k), ("basic", basic), ("advanced", advanced)],
        payload={
            "estimate": basic,
            "epsilon_basic": basic,
            "epsilon_advanced": advanced,
            "advanced_is_tighter": 1 if advanced < basic else 0,
            "delta_prime": dp,
            "k": k,
            "n": k,
            "method": "basic (Thm 3.14) and advanced (Thm 3.20) composition, Dwork & Roth (2014)",
        },
    )


def cheatsheet():
    return "kcompo: composition of differentially private mechanisms"
