# morie.fn -- function file (rootcoder007/morie)
"""Quantile Hadamard sandwich inequality."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kosorok_ch2_quantile_hadamard_inequality"]


def kosorok_ch2_quantile_hadamard_inequality(F, h_n, t_n, p, eps_pn=None):
    r"""Sandwich inequality behind the quantile map's Hadamard
    derivative (Kosorok Ch. 2):

    .. math:: (F + t_n h_n)(\xi^N_{p_n} - \epsilon_{p_n})
              \;\le\; p \;\le\;
              (F + t_n h_n)(\xi^N_{p_n}),

    where :math:`\xi^N_{p_n}` is the perturbed p-quantile. The
    two-sided bracket is what converts a merely monotone inverse into
    a differentiable one: the quantile map has no derivative pointwise,
    but it is pinned between two evaluations of the same perturbed CDF.

    Returns both bounds and whether the target p actually lies between
    them -- a violated sandwich means the perturbed CDF is not a valid
    CDF, which is the failure mode worth catching.

    Parameters
    ----------
    F : callable
        Base CDF.
    h_n : callable
        Perturbation direction.
    t_n : float
        Perturbation scale.
    p : float in (0, 1)
        Quantile level.
    eps_pn : float, optional
        Left offset; a small multiple of t_n by default.

    Returns
    -------
    RichResult
        keys: ``lower``, ``upper``, ``xi_perturbed``, ``p``,
        ``sandwich_holds``, ``eps_pn``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (Hadamard differentiability of the quantile map).
    """
    p = float(p)
    if not 0 < p < 1:
        raise ValueError(f"p must lie in (0, 1), got {p}.")
    t_n = float(t_n)
    if t_n <= 0:
        raise ValueError(f"t_n must be positive, got {t_n}.")
    eps = abs(t_n) if eps_pn is None else float(eps_pn)
    if eps < 0:
        raise ValueError("eps_pn must be non-negative.")

    Fp = lambda z: float(F(z)) + t_n * float(h_n(z))
    # locate the perturbed p-quantile by bisection on the monotone Fp
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Fp(mid) < p:
            lo = mid
        else:
            hi = mid
    xi = 0.5 * (lo + hi)
    lower = Fp(xi - eps)
    upper = Fp(xi)
    return RichResult(
        payload={"lower": float(lower), "upper": float(upper),
                 "xi_perturbed": float(xi), "p": p,
                 "sandwich_holds": bool(lower <= p + 1e-8 <= upper + 1e-8),
                 "eps_pn": eps,
                 "method": "(F + t_n h_n)(xi - eps) <= p <= (F + t_n h_n)(xi)"}
    )


def cheatsheet():
    return "ksr043: two-sided bracket makes the monotone inverse differentiable"
