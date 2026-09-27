"""Expectation of the geometric distribution: sum k (1-p)^(k-1) p = 1/p.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (4.77)-(4.80).
"""

from ._richresult import RichResult

__all__ = ["geomexp"]


def geomexp(p, terms=2000):
    """Expected waiting time (number of iterations) of the geometric distribution.

    The series 1 p + 2 (1-p) p + 3 (1-p)^2 p + ... (4.77) is summed directly
    (``terms`` terms) and by the book's rearrangement into geometric series
    (4.78)-(4.79): row j sums to (1-p)^j, so the total is
    1 + (1-p) + (1-p)^2 + ... = 1/p.

    Parameters
    ----------
    p : float
        Success probability per iteration, 0 < p <= 1.
    terms : int
        Number of series terms.

    Returns
    -------
    RichResult
        Keys: mean (1/p), series, row_sums (the rearranged rows summed).

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (4.77)-(4.80).

    Examples
    --------
    >>> round(geomexp(0.25)["series"], 12)
    4.0
    """
    p = float(p)
    if not 0 < p <= 1 or terms < 1:
        raise ValueError("need 0 < p <= 1 and terms >= 1")
    q = 1 - p
    series = sum(k * q ** (k - 1) * p for k in range(1, terms + 1))
    rows = sum(q**j for j in range(terms))
    return RichResult(
        title="Expectation of the geometric distribution",
        summary_lines=[("1/p", 1 / p), ("series", series)],
        payload={"mean": 1 / p, "series": series, "row_sums": rows},
    )


def cheatsheet():
    return "geomexp: geometric expectation sum k q^(k-1) p = 1/p. Morin (2016) eqs (4.77)-(4.80)."
