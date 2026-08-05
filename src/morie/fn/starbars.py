"""Stars and bars: N_U_n = C(n + N - 1, N - 1) unordered picks with repetition.

Morin (2016), Probability: For the Enthusiastic Beginner,
eqs (1.16)-(1.17), (1.48)-(1.50), (1.53).
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["starbars"]


def starbars(n, N):
    """Unordered picks of n objects from N types, repetition allowed.

    Eq (1.16) is the stars-and-bars count C(n + (N-1), N-1); the equal
    form C(n + N - 1, n) is computed as a cross-check.  The worked cases
    are eq (1.17) (n=10, N=4 -> 286), eq (1.48) (n=2, N=6 -> 21),
    eq (1.49) (n=2 -> N(N+1)/2), eq (1.50) (N=2 -> n+1) and eq (1.53)
    (N=3 -> (n+1)(n+2)/2).

    Parameters
    ----------
    n : int
        Number of picks.
    N : int
        Number of distinct types, N >= 1.

    Returns
    -------
    RichResult
        Keys: n_picks, n_types, count, count_alt, forms_agree.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs. (1.16)-(1.17), (1.48)-(1.50), (1.53).
    """
    count = _morin.stars_and_bars(n, N)
    alt = _morin.binom(int(n) + int(N) - 1, int(n))
    if count != alt:
        raise AssertionError("the two stars-and-bars forms disagree")
    payload = {"n_picks": float(n), "n_types": float(N),
               "count": float(count), "count_alt": float(alt),
               "forms_agree": 1.0}
    return RichResult(
        title="Stars and bars: N_U_n = C(n + N - 1, N - 1).",
        summary_lines=[("n", n), ("N", N), ("count", count)],
        payload=payload,
    )


def cheatsheet():
    return "starbars: N_U_n = C(n+N-1, N-1) unordered picks with repetition. Morin (2016) eq (1.16)."
