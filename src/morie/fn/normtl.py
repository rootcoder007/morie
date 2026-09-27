"""Tolerance factors for normally distributed populations."""

import math

from ._richresult import RichResult
from ._stats_core import chi2, nct, norm

__all__ = ["normal_tolerance_factor"]


def normal_tolerance_factor(n, coverage=0.95, confidence=0.95, sided="two"):
    r"""Tolerance factor k for :math:`\bar x \pm k s` covering a proportion of N(mu, sigma^2).

    Hedderich, Sachs & Reynarowych (2023, Sec. 6.17, eqs 6.174-6.175):
    with confidence :math:`\gamma` at least a proportion ``coverage`` of the
    population lies below :math:`\bar x + k_1 s` when

    .. math::  k_1 = t'_{n-1,\gamma}(z_{coverage}\sqrt n)/\sqrt n,

    the :math:`\gamma` quantile of the noncentral t (exact). Two-sided limits
    :math:`\bar x \pm k_2 s` use Howe's (1969) approximation

    .. math::

        k_2 = \sqrt{(n-1)(1 + 1/n)\,z^2_{(1-coverage)/2}\,/\,\chi^2_{1-\gamma, n-1}}.

    Parameters
    ----------
    n : int
        Sample size, at least 2.
    coverage : float
        Proportion of the population to be covered.
    confidence : float
        Confidence :math:`\gamma`.
    sided : str
        ``"one"`` or ``"two"``.

    Returns
    -------
    RichResult
        ``k``, ``n``, ``coverage``, ``confidence``, ``sided``.

    References
    ----------
    Howe, W. G. (1969). Two-sided tolerance limits for normal populations --
    some improvements. JASA 64, 610-620. Hedderich, J., Sachs, L. &
    Reynarowych, Z. (2023). Applied Statistics: Methods Using R. Springer,
    eqs (6.174)-(6.175), Table 6.21.
    """
    n = int(n)
    if n < 2 or not 0 < coverage < 1 or not 0 < confidence < 1:
        raise ValueError("need n >= 2 and coverage, confidence in (0, 1)")
    if sided == "one":
        k = float(nct.ppf(confidence, n - 1, float(norm.ppf(coverage)) * math.sqrt(n))) / math.sqrt(n)
    elif sided == "two":
        z = float(norm.ppf((1 - coverage) / 2))
        k = math.sqrt((n - 1) * (1 + 1 / n) * z * z / float(chi2.ppf(1 - confidence, n - 1)))
    else:
        raise ValueError("`sided` must be 'one' or 'two'")
    return RichResult(
        title="Normal tolerance factor",
        summary_lines=[("k", k), ("n", n), ("sided", sided)],
        payload={"k": k, "n": n, "coverage": coverage, "confidence": confidence, "sided": sided},
    )


def cheatsheet():
    return "normtl: tolerance factor k (one-sided noncentral t, two-sided Howe) for xbar +/- k s"
