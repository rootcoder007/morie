# morie.fn -- slice s03 (rootcoder007/morie)
"""TMLE with a bounded clever covariate.

Source consulted: van der Laan, M. J. and Rubin, D. (2006), *The
International Journal of Biostatistics* 2(1), article 11, for the
targeting step; and Petersen, M. L., Porter, K. E., Gruber, S., Wang, Y.
and van der Laan, M. J. (2012).  Diagnosing and responding to violations
in the positivity assumption.  *Statistical Methods in Medical Research*
21(1), 31-54, for the response to a near-zero propensity score: bound
the estimated g away from 0 and 1 at a level chosen in advance, which
bounds the clever covariate

    H(D, X) = D / g(X) - (1 - D) / (1 - g(X))

and so bounds the influence of any single observation.  Neither source
was retrievable here as a full text; the truncation rule is quoted in
its standard published form.  Truncation trades a bounded variance for a
bias that does not vanish, so both the bound actually binding and the
number of observations it touched are reported -- the diagnostic
Petersen et al. insist on, not a silent fix.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["tmle_robust"]


def tmle_robust(y, D, X=None, trim=0.025, alpha=0.05):
    """Targeted ATE with the propensity score truncated at ``trim``.

    Returns
    -------
    RichResult with payload:
        estimate  : the targeted ATE
        se, ci_lo, ci_hi
        n_trimmed : observations whose g was bounded
        min_g, max_g : the propensity range before truncation
        psi_untrimmed : the same estimator without truncation
    """
    fit = k.tmle_ate(y, D, X, float(trim))
    raw = k.tmle_ate(y, D, X, 0.0)
    g0 = raw["g"]
    t = float(trim)
    ntr = 0
    for v in g0:
        if v < t or v > 1.0 - t:
            ntr += 1
    z = k.qnorm(1.0 - float(alpha) / 2.0)
    return RichResult(
        title="Robust TMLE (bounded clever covariate)",
        summary_lines=[("ATE", fit["psi"]), ("trimmed", ntr)],
        payload={
            "estimate": fit["psi"],
            "se": fit["se"],
            "ci_lo": fit["psi"] - z * fit["se"],
            "ci_hi": fit["psi"] + z * fit["se"],
            "n_trimmed": ntr,
            "min_g": min(g0) if g0 else float("nan"),
            "max_g": max(g0) if g0 else float("nan"),
            "psi_untrimmed": raw["psi"],
            "eps": fit["eps"],
            "trim": t,
            "n": len(g0),
            "method": "TMLE with the propensity score bounded (Petersen et al. 2012 positivity rule)",
        },
    )


def cheatsheet():
    return "tmlrbt: Robust TMLE under model misspecification"


tmlerobust = tmle_robust
