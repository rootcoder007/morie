# morie.fn -- function file (rootcoder007/morie)
"""Quantile Taylor bounds."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_ch2_quantile_taylor_bounds"]


def kosorok_ch2_quantile_taylor_bounds(F, h, t_n, p, eps_pn=None):
    r"""First-order bounds on the perturbed quantile (Kosorok Ch. 2):

    .. math:: F(\xi^N_{p_n}) + t_n h(\xi_{p_n} - \epsilon_{p_n})
              + o(t_n) \;\le\; p \;\le\;
              F(\xi^N_{p_n}) + t_n h(\xi^N_{p_n}) + o(t_n).

    Expanding :mod:`morie.fn.ksr043`'s sandwich to first order in
    :math:`t_n`. As :math:`t_n \downarrow 0` the two sides collapse
    onto each other at rate :math:`o(t_n)`, and the common limit is
    the Hadamard derivative :math:`-h(\xi_p)/f(\xi_p)` -- which is
    returned so the collapse can be checked, not asserted.

    Parameters
    ----------
    F : callable
        Base CDF.
    h : callable
        Direction.
    t_n : float
        Scale.
    p : float in (0, 1)
        Level.
    eps_pn : float, optional
        Left offset.

    Returns
    -------
    RichResult
        keys: ``lower``, ``upper``, ``gap`` (upper - lower),
        ``implied_derivative``, ``xi_p``, ``t_n``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (quantile expansions).
    """
    p = float(p)
    if not 0 < p < 1:
        raise ValueError(f"p must lie in (0, 1), got {p}.")
    t_n = float(t_n)
    if t_n <= 0:
        raise ValueError(f"t_n must be positive, got {t_n}.")
    eps = abs(t_n) if eps_pn is None else float(eps_pn)

    def quantile(fn, level):
        lo, hi = -50.0, 50.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if fn(mid) < level:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    xi_p = quantile(lambda z: float(F(z)), p)
    Fp = lambda z: float(F(z)) + t_n * float(h(z))
    xi_n = quantile(Fp, p)
    lower = float(F(xi_n)) + t_n * float(h(xi_p - eps))
    upper = float(F(xi_n)) + t_n * float(h(xi_n))
    # density by central difference, for the implied derivative
    d = 1e-5
    dens = (float(F(xi_p + d)) - float(F(xi_p - d))) / (2 * d)
    implied = -float(h(xi_p)) / dens if dens > 0 else np.nan
    return RichResult(
        payload={"lower": lower, "upper": upper, "gap": upper - lower,
                 "implied_derivative": implied, "xi_p": xi_p,
                 "xi_perturbed": xi_n, "t_n": t_n,
                 "method": "First-order quantile bounds; limit is -h(xi_p)/f(xi_p)"}
    )


def cheatsheet():
    return "ksr044: bounds collapse at o(t_n) onto -h(xi_p)/f(xi_p)"
