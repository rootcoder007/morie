# morie.fn -- function file (rootcoder007/morie)
"""Bracketing entropy integral."""

import numpy as np

from scipy import integrate

from ._richresult import RichResult

__all__ = ["kosorok_ch2_donsker_bracketing_integral"]


def kosorok_ch2_donsker_bracketing_integral(N_bracket, delta=1.0, r=2, F=None, P=None):
    r"""Bracketing entropy integral

    .. math:: J_{[\,]}(\delta, \mathcal F, L_r(P))
              = \int_0^\delta
                \sqrt{\log N_{[\,]}(\epsilon, \mathcal F, L_r(P))}
                \, d\epsilon.

    The square root is what makes the integral converge for classes
    whose bracketing numbers blow up polynomially: :math:`\log` of a
    polynomial is logarithmic, and :math:`\sqrt{\log(1/\epsilon)}`
    is integrable at 0. A class with exponentially growing brackets
    diverges -- which is exactly the Donsker boundary.

    Parameters
    ----------
    N_bracket : callable
        eps -> bracketing number.
    delta : float, default 1.0
        Upper limit.
    r : int, default 2
        Interface compatibility (the exponent lives in N_bracket).
    F, P : ignored
        Interface compatibility.

    Returns
    -------
    RichResult
        keys: ``J``, ``finite``, ``delta``, ``abs_error``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (bracketing entropy integral).
    """
    delta = float(delta)
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}.")

    def integrand(eps):
        n = float(N_bracket(eps))
        if n < 1:
            return 0.0
        return float(np.sqrt(np.log(n)))

    val, err = integrate.quad(integrand, 1e-10, delta, limit=200)
    return RichResult(
        payload={"J": float(val), "finite": bool(np.isfinite(val)),
                 "delta": delta, "abs_error": float(err),
                 "method": "J_[](delta) = int_0^delta sqrt(log N_[](eps)) deps"}
    )


def cheatsheet():
    return "ksr035: sqrt(log N) integrable at 0 for polynomial N; that IS the boundary"
