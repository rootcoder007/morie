# morie.fn -- function file (rootcoder007/morie)
"""Minimum sample size for logistic regression."""

import math

from . import _stats_core as stats


def sample_size_logistic(
    p0: float,
    p1: float | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
    *,
    two_sided: bool = True,
    B: float = 0.5,
    covariate: str = "binary",
    odds_ratio: float | None = None,
) -> int:
    r"""Total sample size for logistic regression (Hsieh, Bloch & Larsen 1998).

    Binary covariate (their eq. 1): with event rates :math:`P_0` at
    :math:`X = 0` and :math:`P_1` at :math:`X = 1`, a fraction :math:`B` of
    the sample at :math:`X = 1` and :math:`\bar P = (1-B)P_0 + BP_1`,

    .. math::

        n = \frac{\left[z_{1-\alpha/2}\sqrt{\bar P(1-\bar P)/B}
            + z_{1-\beta}\sqrt{P_0(1-P_0) + P_1(1-P_1)(1-B)/B}\right]^2}
            {(P_0 - P_1)^2 (1-B)}.

    Continuous, normally distributed covariate (their eq. 2):
    :math:`n = (z_{1-\alpha/2} + z_{1-\beta})^2 / (P_0(1-P_0)\beta^{*2})`,
    with :math:`P_0` the event rate at the covariate mean and
    :math:`\beta^* = \log` (odds ratio per standard deviation).

    Parameters
    ----------
    p0 : float
        Event rate at X = 0 (binary) or at the covariate mean (continuous).
    p1 : float, optional
        Event rate at X = 1 (binary covariate). Give either ``p1`` or
        ``odds_ratio``.
    alpha, power : float
        Type I error and power.
    two_sided : bool
        Two-sided test (``z_{1-alpha/2}``) or one-sided (``z_{1-alpha}``).
    B : float
        Fraction of the sample with X = 1 (binary covariate).
    covariate : {"binary", "continuous"}
    odds_ratio : float, optional
        Odds ratio for X = 1 vs 0 (binary) or per standard deviation
        (continuous).

    Returns
    -------
    int
        Total sample size, rounded up.

    References
    ----------
    Hsieh, F. Y., Bloch, D. A., & Larsen, M. D. (1998). A simple method of
    sample size calculation for linear and logistic regression. Statistics
    in Medicine, 17(14), 1623-1634. (R: powerMediation::SSizeLogisticBin,
    SSizeLogisticCon.)

    Examples
    --------
    >>> sample_size_logistic(0.2, 0.35)
    276
    >>> sample_size_logistic(0.2, covariate="continuous", odds_ratio=1.5)
    299
    """
    if not 0 < p0 < 1:
        raise ValueError(f"p0 must be in (0, 1), got {p0}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1), got {power}.")
    za = float(stats.norm.ppf(1 - (alpha / 2 if two_sided else alpha)))
    zb = float(stats.norm.ppf(power))
    if covariate == "continuous":
        if odds_ratio is None or odds_ratio <= 0 or odds_ratio == 1:
            raise ValueError("continuous covariate: odds_ratio > 0 and != 1 is required.")
        return int(math.ceil((za + zb) ** 2 / (p0 * (1 - p0) * math.log(odds_ratio) ** 2)))
    if covariate != "binary":
        raise ValueError("covariate must be 'binary' or 'continuous'.")
    if p1 is None:
        if odds_ratio is None or odds_ratio <= 0:
            raise ValueError("give p1 or a positive odds_ratio.")
        p1 = odds_ratio * p0 / (1 - p0 + odds_ratio * p0)
    if not 0 < p1 < 1 or p1 == p0:
        raise ValueError(f"p1 must be in (0, 1) and differ from p0, got {p1}.")
    if not 0 < B < 1:
        raise ValueError(f"B must be in (0, 1), got {B}.")
    pbar = (1 - B) * p0 + B * p1
    num = (za * math.sqrt(pbar * (1 - pbar) / B) + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1) * (1 - B) / B)) ** 2
    return int(math.ceil(num / ((p0 - p1) ** 2 * (1 - B))))


n_logit = sample_size_logistic


def cheatsheet() -> str:
    return "sample_size_logistic({}) -> Minimum sample size for logistic regression."
