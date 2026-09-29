# morie.fn -- function file (rootcoder007/morie)
"""Power for two-proportion z-test."""

from __future__ import annotations

from . import _powercore as pc


def power_prop_test(
    n: float | None = None,
    p1: float | None = None,
    p2: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    *,
    alternative: str = "two-sided",
    strict: bool = True,
    method: str = "fleiss",
) -> float:
    r"""Power for two-proportion z-test.

    Solve for ``n`` (per group) or ``power`` of the two-sample test of
    proportions. ``method="fleiss"`` (default) is the normal approximation
    of R's ``power.prop.test`` (Fleiss 1981): ``power = Phi((sqrt(n) |p1 -
    p2| - z sqrt(2 pbar qbar)) / sqrt(p1 q1 + p2 q2))`` with ``z =
    z_{1 - alpha/tside}``, plus the far tail when ``strict`` and two-sided
    (``strict=False`` is R's default). ``method="cohen_h"`` uses Cohen's
    (1988) arcsine effect size ``h = |2 asin sqrt(p1) - 2 asin sqrt(p2)|``
    with ``power = Phi(h sqrt(n/2) - z) (+ Phi(-h sqrt(n/2) - z))``, as
    ``pwr::pwr.2p.test``. ``n`` is found by bisection on ``[1, 1e7]``.

    :param n: Sample size per group.
    :param p1: Proportion in group 1.
    :param p2: Proportion in group 2.
    :param alpha: Significance level.
    :param power: Target power.
    :param alternative: ``"two-sided"`` or ``"one-sided"``.
    :param strict: Include the far rejection tail (``fleiss``).
    :param method: ``"fleiss"`` or ``"cohen_h"``.
    :return: ``n`` or ``power``.

    References
    ----------
    Fleiss, J. L. (1981). *Statistical Methods for Rates and Proportions*, 2nd ed. Wiley.

    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Erlbaum, ch. 6.

    Examples
    --------
    >>> round(power_prop_test(n=100, p1=0.5, p2=0.7), 12)
    0.828109771966
    >>> round(power_prop_test(p1=0.5, p2=0.7, power=0.8), 8)
    92.99869757
    """
    if p1 is None or p2 is None:
        raise ValueError("p1 and p2 must both be provided.")
    for v, nm in ((p1, "p1"), (p2, "p2")):
        if not 0 < v < 1:
            raise ValueError(f"{nm} must be in (0, 1), got {v}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if (n is None) == (power is None):
        raise ValueError("Provide exactly one of (n, power) when p1 and p2 are given.")
    tside = {"two-sided": 2, "one-sided": 1, "greater": 1}.get(alternative)
    if tside is None:
        raise ValueError(f"alternative must be 'two-sided' or 'one-sided', got {alternative!r}.")
    if method == "fleiss":

        def pw(nn):
            return pc.prop_power(float(nn), float(p1), float(p2), float(alpha), tside, strict)

    elif method == "cohen_h":

        def pw(nn):
            return pc.cohen_h_power(float(nn), float(p1), float(p2), float(alpha), tside)

    else:
        raise ValueError("method must be 'fleiss' or 'cohen_h'")
    if power is None:
        return pw(n)
    return pc.solve_up(lambda v: pw(v) - power, 1.0, 1e7)


pwr_p = power_prop_test


def cheatsheet() -> str:
    return "power_prop_test(n, p1, p2, alpha, power) -> power.prop.test (Fleiss) or Cohen's h"


# compact alias per ledger/NAMING.md
powerproptest = power_prop_test
