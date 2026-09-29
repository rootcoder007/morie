# morie.fn -- function file (rootcoder007/morie)
"""Power analysis for t-tests."""

from __future__ import annotations

from . import _powercore as pc


def power_t_test(
    n: float | None = None,
    delta: float | None = None,
    sd: float = 1.0,
    alpha: float = 0.05,
    power: float | None = None,
    *,
    alternative: str = "two-sided",
    type: str = "two-sample",
    strict: bool = True,
) -> float:
    r"""Power analysis for t-tests.

    Solve for whichever one of ``n``, ``delta`` and ``power`` is ``None``,
    as R's ``power.t.test``: with ``nu = (n - 1) tsample`` (``tsample = 2``
    for two independent samples, else 1) and ``ncp = sqrt(n / tsample)
    delta / sd``, the power is ``P(T'(nu, ncp) > t_{1 - alpha/tside, nu})``
    (Cohen 1988, ch. 2). With ``strict=True`` (default, the exact rejection
    probability of the two-sided test) the lower rejection tail ``P(T' <
    -t)`` is added; ``strict=False`` drops it, reproducing R's default.
    ``n`` and ``delta`` are found by bisection to ``1e-12``, the upper
    bracket doubled from the lower end of R's ranges ``[2, 1e7]`` and ``sd
    [1e-7, 1e7]``.

    :param n: Observations per group (two-sample) or pairs / observations.
    :param delta: True difference in means (raw scale).
    :param sd: Standard deviation.
    :param alpha: Significance level.
    :param power: Target power.
    :param alternative: ``"two-sided"`` or ``"one-sided"``.
    :param type: ``"two-sample"``, ``"one-sample"`` or ``"paired"``.
    :param strict: Include the far rejection tail of a two-sided test.
    :return: The missing quantity.

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Erlbaum, ch. 2.

    R Core Team. ``power.t.test`` (package stats).

    Examples
    --------
    >>> round(power_t_test(n=20, delta=1.0), 12)
    0.868953027724
    >>> round(power_t_test(delta=1.0, power=0.9), 8)
    22.02108843
    """
    if sum(v is None for v in (n, delta, power)) != 1:
        raise ValueError("Exactly one of n, delta, or power must be None.")
    if sd <= 0:
        raise ValueError(f"sd must be > 0, got {sd}.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if power is not None and not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1), got {power}.")
    tside = {"two-sided": 2, "one-sided": 1, "greater": 1}.get(alternative)
    if tside is None:
        raise ValueError(f"alternative must be 'two-sided' or 'one-sided', got {alternative!r}.")
    tsample = {"two-sample": 2, "one-sample": 1, "paired": 1}.get(type)
    if tsample is None:
        raise ValueError(f"type must be 'two-sample', 'one-sample' or 'paired', got {type!r}.")
    if delta is not None and tside == 2:
        delta = abs(delta)

    def pw(nn, dd):
        return pc.t_power(float(nn), float(dd), float(sd), float(alpha), tsample, tside, strict)

    if power is None:
        return pw(n, delta)
    if n is None:
        return pc.solve_up(lambda v: pw(v, delta) - power, 2.0, 1e7)
    return pc.solve_up(lambda v: pw(n, v) - power, sd * 1e-7, sd * 1e7)


pwr_t = power_t_test


def cheatsheet() -> str:
    return "power_t_test(n, delta, sd, alpha, power) -> power.t.test (strict two-sided by default)"


# compact alias per ledger/NAMING.md
powerttest = power_t_test
