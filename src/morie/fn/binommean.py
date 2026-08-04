"""Binomial mean E(k) = np, verified against the pmf series.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.61).
"""

from . import _array_core as np

from . import _morin

from ._richresult import RichResult

__all__ = ["binommean"]


def binommean(n, p):
    """Binomial mean E(k) = np, verified against the pmf series.

    Parameters
    ----------
    n : int
        Trials, >= 0.
    p : float
        Success probability, in [0, 1].

    Returns
    -------
    RichResult
        Keys: mean, series_mean.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.61).
    """
    value = _morin.binomial_mean(n, p)
    pmf = _morin.binomial_pmf_vector(n, p)
    series = float(np.sum(np.arange(int(n) + 1) * pmf))
    if abs(series - value) > 1e-9 * max(1.0, value):
        raise AssertionError("series mean disagrees with np")
    payload = {"mean": value, "series_mean": series}
    lines = [("np", value)]
    return RichResult(
        title="Binomial mean E(k) = np, verified against the pmf series.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "binommean: Binomial mean np. Morin (2016) eq (4.61)."
