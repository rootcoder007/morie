# morie.fn -- function file (rootcoder007/morie)
"""Uno C-index for censored data."""

from .cstat import cstat

from ._richresult import RichResult

__all__ = ["uno_concordance"]


def uno_concordance(time, event, predicted_risk):
    """Truncated, IPCW-weighted concordance index.

    Harrell's C weights every comparable pair equally, which makes its
    value depend on the censoring distribution of the particular study:
    the same model scores differently under heavier censoring.  Uno's
    version reweights each pair by the inverse squared censoring survival
    at the earlier event time and truncates at the 75th percentile of the
    event times, which removes that dependence.  The estimator already
    exists as ``cstat(..., method="uno")``, so this is a thin alias.

    Formula: truncated Harrell C with IPCW weights ``1/G(T_i)^2``.

    Parameters
    ----------
    time : array-like
        Observed event or censoring times.
    event : array-like
        Event indicator, 1 = event, 0 = censored.
    predicted_risk : array-like
        Predicted risk; higher means shorter expected survival.

    Returns
    -------
    RichResult
        ``estimate``, ``c_statistic``, ``se``, ``ci_lower``, ``ci_upper``,
        ``concordant``, ``discordant``, ``tied``, ``comparable``,
        ``method``.

    References
    ----------
    Uno, H., Cai, T., Pencina, M. J., D'Agostino, R. B. & Wei, L. J.
    (2011).  On the C-statistics for evaluating overall adequacy of risk
    prediction procedures with censored survival data.  Statistics in
    Medicine 30(10):1105-1117.  <https://doi.org/10.1002/sim.4154>
    """
    r = cstat(time, event, predicted_risk, method="uno")
    return RichResult(payload={
        "estimate": float(r["c_statistic"]), "c_statistic": float(r["c_statistic"]),
        "se": float(r["se"]), "ci_lower": float(r["ci_lower"]),
        "ci_upper": float(r["ci_upper"]), "concordant": int(r["concordant"]),
        "discordant": int(r["discordant"]), "tied": int(r["tied"]),
        "comparable": int(r["comparable"]),
        "method": "Uno IPCW-weighted truncated C-statistic [Uno et al. 2011]"})


# CANONICAL TEST
# >>> # risk perfectly ordered against survival: C is exactly 1
# >>> r = uno_concordance([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0], [4.0, 3.0, 2.0, 1.0])
# >>> assert abs(r["estimate"] - 1.0) < 1e-12
# >>> # reversing the risk score sends it to exactly 0
# >>> q = uno_concordance([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0])
# >>> assert abs(q["estimate"]) < 1e-12


def cheatsheet():
    return "survci2(time, event, predicted_risk): Uno C (alias of cstat)."
