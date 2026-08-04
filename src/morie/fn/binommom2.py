"""Binomial second moment E(k^2) = p^2 n(n-1) + pn.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.66).
"""

from . import _array_core as np

from . import _morin

from ._richresult import RichResult

__all__ = ["binommom2"]


def binommom2(n, p):
    """Binomial second moment E(k^2) = p^2 n(n-1) + pn.

    Parameters
    ----------
    n : int
        Trials, >= 0.
    p : float
        Success probability, in [0, 1].

    Returns
    -------
    RichResult
        Keys: second_moment.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.66).
    """
    value = _morin.binomial_second_moment(n, p)
    pmf = _morin.binomial_pmf_vector(n, p)
    ks = np.arange(int(n) + 1)
    series = float(np.sum(ks ** 2 * pmf))
    if abs(series - value) > 1e-9 * max(1.0, value):
        raise AssertionError("series second moment disagrees")
    payload = {"second_moment": value}
    lines = [("E(k^2)", value)]
    return RichResult(
        title="Binomial second moment E(k^2) = p^2 n(n-1) + pn.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "binommom2: Binomial second moment p^2 n(n-1) + pn. Morin (2016) eq (4.66)."
