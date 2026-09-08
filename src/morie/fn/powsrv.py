# morie.fn -- function file (rootcoder007/morie)
"""Power of a two-sided test under a complex survey design."""

import math

from . import _s03core as core

from ._richresult import RichResult

__all__ = ["power_survey"]


def power_survey(effect_size, alpha=0.05, DEFF=1.0, n=100):
    """Power after discounting the sample size by the design effect.

    A clustered design does not reduce the sample size, it reduces the
    INFORMATION: the effective sample size is ``n / DEFF``, and every
    power calculation done on the nominal n is optimistic by exactly that
    factor.  Both tails are kept in the power expression, so a zero
    effect returns alpha exactly rather than alpha/2 -- the standard
    one-tail shortcut is wrong at the null and is a common source of
    off-by-a-factor-of-two power tables.

    Formula: ``n_eff = n / DEFF``; ``power = Phi(d sqrt(n_eff) - z) +
    Phi(-d sqrt(n_eff) - z)`` with ``z = Phi^-1(1 - alpha/2)``.

    Parameters
    ----------
    effect_size : float
        Standardised effect size (Cohen's d).
    alpha : float, default 0.05
        Two-sided significance level, in (0, 1).
    DEFF : float, default 1.0
        Design effect, at least 1 in any real clustered design but any
        positive value is accepted.
    n : int, default 100
        Nominal sample size, at least 1.

    Returns
    -------
    RichResult
        ``estimate`` (the power), ``power``, ``n_eff``, ``ncp`` (the
        non-centrality ``d sqrt(n_eff)``), ``z_crit``.

    References
    ----------
    Lumley, T. (2010).  Complex Surveys: A Guide to Analysis Using R,
    Wiley.  doi:10.1002/9780470580066.  The design-effect discount of
    the effective sample size is the standard form used there; the book
    is not held locally, so the standard form was used rather than a
    rendered page.
    """
    d = float(effect_size)
    alpha = float(alpha)
    deff = float(DEFF)
    n = int(n)
    if not (0.0 < alpha < 1.0):
        raise ValueError("power_survey: alpha must lie in (0, 1)")
    if deff <= 0.0:
        raise ValueError("power_survey: DEFF must be positive")
    if n < 1:
        raise ValueError("power_survey: n must be at least 1")
    neff = n / deff
    z = core.qnorm(1.0 - alpha / 2.0)
    ncp = d * math.sqrt(neff)
    power = core.pnorm(ncp - z) + core.pnorm(-ncp - z)
    return RichResult(payload={
        "estimate": power, "power": power, "n_eff": neff, "ncp": ncp,
        "z_crit": z, "DEFF": deff, "n": n,
        "method": "Two-sided z power with the design effect discount"})


def cheatsheet():
    return "powsrv: Survey-design-aware power"

# public names resolved by fn/_lazy_map.json
powersurvey = power_survey
