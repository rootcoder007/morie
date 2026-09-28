# morie.fn -- function file (rootcoder007/morie)
"""Inference for (multiscale) geographically weighted regression: local t-values against the
da Silva-Fotheringham multiple-testing corrected critical value, and bandwidth confidence intervals from
Akaike weights."""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._richresult import RichResult
from ._rrng_core import qt

__all__ = ["mgwr_local_t", "bandwidth_confidence_interval"]


def mgwr_local_t(coef, se, enp: float, *, alpha: float = 0.05, df: float | None = None) -> RichResult:
    r"""Local t-values ``beta_i / se_i`` of one (M)GWR coefficient surface with the corrected critical value.

    The family-wise error rate over the ``n`` dependent local tests is
    controlled by the Bonferroni-type adjustment of da Silva and Fotheringham
    (2016): ``alpha_adj = alpha / ENP_j`` with ``ENP_j`` the effective number
    of parameters of that surface (in MGWR, the trace of its hat-matrix
    component), and critical value ``t_{1 - alpha_adj/2, df}`` (``df``
    default ``n - ENP_j``).

    References
    ----------
    da Silva, A. R. and Fotheringham, A. S. (2016). The multiple testing issue
    in geographically weighted regression. *Geographical Analysis*, 48, 233-247.
    Yu, H., Fotheringham, A. S., Li, Z., Oshan, T., Kang, W. and Wolf, L. J.
    (2020). Inference in multiscale geographically weighted regression.
    *Geographical Analysis*, 52, 87-106.

    Examples
    --------
    >>> r = mgwr_local_t([0.5, -0.2, 1.1], [0.2, 0.25, 0.3], 2.0, df=50)
    >>> [round(v, 6) for v in r.t], round(r.alpha_adjusted, 6), r.significant
    ([2.5, -0.8, 3.666667], 0.025, [True, False, True])
    """
    b, s = [float(v) for v in coef], [float(v) for v in se]
    n = len(b)
    t = [b[i] / s[i] for i in range(n)]
    a = alpha / enp
    d = n - enp if df is None else df
    crit = qt(1 - a / 2, d)
    return RichResult(
        payload={"t": t, "alpha_adjusted": a, "critical": crit, "significant": [abs(v) > crit for v in t]}
    )


def bandwidth_confidence_interval(bandwidths, aicc, *, level: float = 0.95) -> RichResult:
    r"""Bandwidth confidence interval from Akaike weights over candidate bandwidths (Li et al. 2020).

    ``w_k = exp(-Delta_k / 2) / sum exp(-Delta_j / 2)`` with ``Delta_k`` the
    AICc difference from the best bandwidth; candidates are added in order
    of weight until the cumulative weight reaches ``level``, and the interval
    spans the included bandwidths.

    References
    ----------
    Li, Z., Fotheringham, A. S., Oshan, T. M. and Wolf, L. J. (2020).
    Measuring bandwidth uncertainty in multiscale geographically weighted
    regression using Akaike weights. *Annals of the American Association of
    Geographers*, 110, 1500-1520.
    Burnham, K. P. and Anderson, D. R. (2002). *Model Selection and
    Multimodel Inference*, 2nd edn. Springer, section 4.7.

    Examples
    --------
    >>> r = bandwidth_confidence_interval([40, 50, 60, 70, 80], [310.0, 302.0, 300.0, 301.0, 306.0])
    >>> r.best, r.lower, r.upper
    (60.0, 50.0, 70.0)
    """
    bw, a = [float(v) for v in bandwidths], [float(v) for v in aicc]
    m = min(a)
    e = [math.exp(-(v - m) / 2) for v in a]
    tot = ssum(e)
    w = [v / tot for v in e]
    order = sorted(range(len(bw)), key=lambda k: (-w[k], k))
    cum, inc = 0.0, []
    for k in order:
        inc.append(k)
        cum += w[k]
        if cum >= level:
            break
    sel = [bw[k] for k in inc]
    return RichResult(
        payload={"best": bw[order[0]], "lower": min(sel), "upper": max(sel), "weights": w, "coverage": cum}
    )


def cheatsheet() -> str:
    return "mgwr_local_t / bandwidth_confidence_interval -> inference for (M)GWR."
