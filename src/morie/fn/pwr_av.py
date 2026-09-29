# morie.fn -- function file (rootcoder007/morie)
"""Power for one-way ANOVA."""

from __future__ import annotations

from . import _powercore as pc


def power_anova(
    n: float | None = None,
    k: int | None = None,
    f: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
) -> float:
    r"""Power for one-way ANOVA.

    Power of the F test of equal means in ``k`` groups of ``n`` each with
    Cohen's effect size ``f = sigma_means / sigma`` (Cohen 1988, ch. 8):
    ``ncp = f^2 k n``, ``df = (k - 1, k (n - 1))``, ``power = P(F'(df,
    ncp) > F_{1 - alpha})`` -- ``pwr::pwr.anova.test``, and R's
    ``power.anova.test`` with ``between.var = k f^2 / (k - 1)`` and
    ``within.var = 1``. Solves for whichever of ``n``, ``f``, ``power`` is
    missing by bisection (``n`` on ``[2, 1e7]``), or for the smallest
    integer ``k`` reaching ``power``.

    :param n: Observations per group.
    :param k: Number of groups.
    :param f: Cohen's f.
    :param alpha: Significance level.
    :param power: Target power.
    :return: The missing quantity.

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Erlbaum, ch. 8.

    Examples
    --------
    >>> round(power_anova(n=20, k=3, f=0.25), 12)
    0.374431076256
    >>> round(power_anova(k=3, f=0.25, power=0.8), 8)
    52.39659747
    """
    if sum(v is None for v in (n, k, f, power)) != 1:
        raise ValueError("Exactly one of n, k, f, or power must be None.")
    if k is not None and k < 2:
        raise ValueError(f"k must be >= 2, got {k}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    def pw(nn, kk, ff):
        return pc.anova_power(float(nn) * float(kk), int(kk), float(ff), float(alpha))

    if power is None:
        return pw(n, k, f)
    if n is None:
        return pc.solve_up(lambda v: pw(v, k, f) - power, 1.0 + 1e-9, 1e7)
    if f is None:
        return pc.solve_up(lambda v: pw(n, k, v) - power, 1e-8, 20.0)
    for kk in range(2, 200):
        if pw(n, kk, f) >= power:
            return float(kk)
    raise ValueError("Could not find k in [2, 200) achieving the desired power.")


pwr_av = power_anova


def cheatsheet() -> str:
    return "power_anova(n, k, f, alpha, power) -> one-way ANOVA power with Cohen's f"


# compact alias per ledger/NAMING.md
poweranova = power_anova
