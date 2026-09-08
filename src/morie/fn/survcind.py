# morie.fn -- tail3 batch (rootcoder007/morie)
"""Harrell's concordance index for censored survival data.

Source consulted: Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L. &
Rosati, R.A. (1982). Evaluating the yield of medical tests.  *JAMA* 247(18),
2543-2546.  The c-index is the proportion of usable (comparable) subject
pairs whose predicted risks are ordered the same way as their observed
survival times.  A pair (i, j) is comparable when the earlier of the two
times is an observed event; the pair is concordant when the subject with the
shorter time carries the higher predicted risk, discordant when it carries
the lower, and tied when the risks are equal (counted as half).

    c = (concordant + 0.5 * tied) / comparable
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["survival_concordance"]


def survival_concordance(time, event, predicted_risk):
    """Harrell's c-index.

    Parameters
    ----------
    time : array-like
        Observed follow-up times.
    event : array-like
        1 if the event was observed, 0 if right-censored.
    predicted_risk : array-like
        Predicted risk score; higher means shorter expected survival.

    Returns
    -------
    RichResult
        statistic (c-index), estimate, concordant, discordant, tied,
        comparable, n, method.

    References
    ----------
    Harrell, Califf, Pryor, Lee & Rosati (1982), JAMA 247(18), 2543-2546.
    """
    t = np.atleast_1d(np.asarray(time, dtype=float)).ravel()
    e = np.atleast_1d(np.asarray(event, dtype=float)).ravel()
    r = np.atleast_1d(np.asarray(predicted_risk, dtype=float)).ravel()
    n = int(min(t.size, e.size, r.size))
    conc = 0.0
    disc = 0.0
    tied = 0.0
    comp = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ti = float(t[i])
            tj = float(t[j])
            if ti == tj:
                continue
            if ti < tj:
                lo, hi = i, j
            else:
                lo, hi = j, i
            if float(e[lo]) != 1.0:
                continue
            comp += 1.0
            if float(r[lo]) > float(r[hi]):
                conc += 1.0
            elif float(r[lo]) < float(r[hi]):
                disc += 1.0
            else:
                tied += 1.0
    c = (conc + 0.5 * tied) / comp if comp > 0.0 else float("nan")
    return RichResult(
        payload={
            "statistic": float(c),
            "estimate": float(c),
            "concordant": float(conc),
            "discordant": float(disc),
            "tied": float(tied),
            "comparable": float(comp),
            "n": n,
            "method": "Harrell concordance index (Harrell et al. 1982)",
        }
    )


# CANONICAL TEST
# >>> # perfect ordering: shorter time always has the higher risk
# >>> r = survival_concordance([1.0, 2.0, 3.0], [1, 1, 1], [3.0, 2.0, 1.0])
# >>> assert abs(r["statistic"] - 1.0) < 1e-12
# >>> r2 = survival_concordance([1.0, 2.0, 3.0], [1, 1, 1], [1.0, 2.0, 3.0])
# >>> assert abs(r2["statistic"]) < 1e-12


def cheatsheet():
    return "survcind(time, event, risk): Harrell c-index."
