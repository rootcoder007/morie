"""Stirling-reduced Gaussian form e^(-x^2/n)/sqrt(pi n) against the exact pmf.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (5.4), (5.13).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["gaussapx"]


def gaussapx(x, n):
    """Stirling-reduced Gaussian form e^(-x^2/n)/sqrt(pi n) against the exact pmf.

    Parameters
    ----------
    x : float
        Deviation from n Heads.
    n : int
        Half the number of flips, >= 1.

    Returns
    -------
    RichResult
        Keys: approx, exact, rel_error.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (5.4), (5.13).
    """
    approx = _morin.gaussian_approx_2n(x, n)
    exact = _morin.binomial_centered_pmf(int(round(float(x))), n)
    payload = {
        "approx": approx,
        "exact": exact,
        "rel_error": abs(approx - exact) / max(exact, 1e-300),
    }
    lines = [("PG(x)", approx), ("PB(x)", exact)]
    return RichResult(
        title="Stirling-reduced Gaussian form e^(-x^2/n)/sqrt(pi n) against the exact pmf.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "gaussapx: Gaussian form e^(-x^2/n)/sqrt(pi n) vs the exact centred binomial. Morin (2016) eqs (5.4), (5.13)."
