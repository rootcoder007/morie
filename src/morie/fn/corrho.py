"""Tests of a Pearson correlation against zero or a hypothesised value."""

import math

from ._richresult import RichResult
from ._stats_core import norm
from ._stats_core import t as tdist

__all__ = ["correlation_test"]


def _p(stat, alternative, sf):
    if alternative == "two-sided":
        return min(1.0, 2.0 * sf(abs(stat)))
    if alternative == "greater":
        return sf(stat)
    if alternative == "less":
        return sf(-stat)
    raise ValueError("`alternative` must be 'two-sided', 'greater' or 'less'")


def correlation_test(r, n, rho0=0.0, alternative="two-sided", method="fisher"):
    r"""Test :math:`H_0: \rho = \rho_0` for a Pearson correlation r from n pairs.

    Hedderich, Sachs & Reynarowych (2023, Sec. 7.8.1):

    * :math:`\rho_0 = 0` (7.389): :math:`\hat t = r\sqrt{(n-2)/(1-r^2)}` on
      :math:`n-2` degrees of freedom, as ``cor.test``.
    * ``method="fisher"`` (7.399): :math:`\hat z = (\tanh^{-1}r - \tanh^{-1}\rho_0)
      \sqrt{n-3}`, standard normal.
    * ``method="samiuddin"`` (7.393): :math:`\hat t = (r-\rho_0)\sqrt{n-2}\,/
      \sqrt{(1-r^2)(1-\rho_0^2)}` on :math:`n-2` degrees of freedom
      (Samiuddin 1970).

    Parameters
    ----------
    r : float
        Sample correlation, in (-1, 1).
    n : int
        Number of pairs.
    rho0 : float
        Hypothesised correlation.
    alternative : str
        ``"two-sided"``, ``"greater"`` or ``"less"``.
    method : str
        ``"fisher"`` or ``"samiuddin"`` (ignored when ``rho0 == 0``).

    Returns
    -------
    RichResult
        ``statistic``, ``p_value``, ``distribution`` (``"t"`` or ``"normal"``),
        ``df``, ``method``.

    References
    ----------
    Fisher, R. A. (1921). On the "probable error" of a coefficient of
    correlation deduced from a small sample. Metron 1, 3-32. Samiuddin, M.
    (1970). On a test for an assigned value of correlation in a bivariate
    normal distribution. Biometrika 57, 461-464. Hedderich, J., Sachs, L. &
    Reynarowych, Z. (2023). Applied Statistics: Methods Using R. Springer,
    eqs (7.389), (7.393), (7.399).
    """
    r, n, rho0 = float(r), int(n), float(rho0)
    if not -1 < r < 1 or not -1 < rho0 < 1 or n < 4:
        raise ValueError("need -1 < r, rho0 < 1 and n >= 4")
    if rho0 == 0.0:
        stat = r * math.sqrt((n - 2) / (1 - r * r))
        df, dist, how = n - 2, "t", "t test of rho = 0"
        p = _p(stat, alternative, lambda v: float(tdist.sf(v, df)))
    elif method == "fisher":
        stat = (math.atanh(r) - math.atanh(rho0)) * math.sqrt(n - 3)
        df, dist, how = None, "normal", "Fisher z test"
        p = _p(stat, alternative, lambda v: float(norm.sf(v)))
    elif method == "samiuddin":
        stat = (r - rho0) * math.sqrt(n - 2) / math.sqrt((1 - r * r) * (1 - rho0 * rho0))
        df, dist, how = n - 2, "t", "Samiuddin t test"
        p = _p(stat, alternative, lambda v: float(tdist.sf(v, df)))
    else:
        raise ValueError("`method` must be 'fisher' or 'samiuddin'")
    return RichResult(
        title="Test of a correlation coefficient",
        summary_lines=[("method", how), ("statistic", stat), ("p-value", p)],
        payload={
            "statistic": stat,
            "p_value": p,
            "distribution": dist,
            "df": df,
            "method": how,
            "r": r,
            "n": n,
            "rho0": rho0,
        },
    )


def cheatsheet():
    return "corrho: r vs rho0 -- t (rho0 = 0), Fisher z or Samiuddin t"
