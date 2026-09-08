# morie.fn -- function file (rootcoder007/morie)
"""Asymptotic normality of the boundary-free KDFE (Theorem 5.3)."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["bfkdfnorm", "fauzi_thm5_3_bdfree_normality"]


def bfkdfnorm(estimate, variance, null=None, bias=0.0, level=0.95):
    r"""Asymptotic normality of the boundary-free KDFE (Theorem 5.3).

    Theorem 5.3: under D1-D5,

    .. math:: \frac{\tilde F_X(x) - F_X(x)}
              {\sqrt{\mathrm{Var}[\tilde F_X(x)]}}
              \;\to_D\; N(0,1).

    The proof is a Lyapunov argument, and it is unusually easy for a
    reason worth keeping: because :math:`0 \le W(v) \le 1` for every
    :math:`v`, the :math:`(2+\delta)` moment of each summand is bounded by
    :math:`2^{2+\delta} < \infty` with NO assumption on :math:`F_X`. The
    estimator is an average of bounded variables, so Lyapunov's condition
    is automatic.

    This routine returns the standardised statistic and a Wald interval
    for :math:`F_X(x)`. The interval is clipped to :math:`[0,1]`, because
    the estimand is a probability and an unclipped Wald interval routinely
    leaves the unit interval near the boundary -- which is precisely the
    region this whole construction exists to handle.

    Note that the CENTRING is :math:`F_X(x)`, not
    :math:`E[\tilde F_X(x)]`. The theorem standardises by the variance
    alone, so the interval inherits the :math:`O(h^2)` bias of
    Theorem 5.2 and is not bias-corrected; ``bias`` may be supplied to
    subtract it.

    Parameters
    ----------
    estimate : float
        ``tilde F_X(x)``.
    variance : float
        ``Var[tilde F_X(x)]``, e.g. from
        :func:`morie.fn.fzt52.bfkdfbv`.
    null : float, optional
        The value of ``F_X(x)`` to test against.
    bias : float, default 0.0
        Bias to subtract before standardising.
    level : float, default 0.95
        Confidence level for the interval.

    Returns
    -------
    RichResult
        Keys ``statistic``, ``p_value``, ``lower``, ``upper``, ``se``, ``level``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Theorem 5.3.
    """
    from . import _stats_core as stats

    var = float(variance)
    if var <= 0:
        raise ValueError(f"the variance must be positive, got {var}.")
    lvl = float(level)
    if not 0.0 < lvl < 1.0:
        raise ValueError(f"level must lie strictly in (0, 1), got {lvl}.")
    se = float(np.sqrt(var))
    centre = float(estimate) - float(bias)
    z = float(stats.norm.ppf(0.5 + lvl / 2.0))
    lower = max(0.0, centre - z * se)
    upper = min(1.0, centre + z * se)
    if null is None:
        stat = float("nan")
        pval = float("nan")
    else:
        stat = (centre - float(null)) / se
        pval = float(2.0 * (1.0 - stats.norm.cdf(abs(stat))))
    return RichResult(
        payload={
            "statistic": float(stat),
            "p_value": float(pval),
            "lower": float(lower),
            "upper": float(upper),
            "se": se,
            "level": lvl,
            "method": "asymptotic normality of the boundary-free KDFE (Theorem 5.3)",
        }
    )


fauzi_thm5_3_bdfree_normality = bfkdfnorm


def cheatsheet():
    return "fzt53: Thm 5.3: normality is automatic because W is bounded in [0,1]; interval clipped to [0,1]"


# CANONICAL TEST
# >>> r = bfkdfnorm(estimate=0.5, variance=0.0025, null=0.5)
# >>> abs(r['statistic']) < 1e-15 and abs(r['p_value'] - 1.0) < 1e-12
# True
