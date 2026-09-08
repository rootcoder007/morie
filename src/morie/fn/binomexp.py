"""Binomial expansion of (x + d)^n, term by term with the remainder.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (7.35).
"""

import math

from ._richresult import RichResult

__all__ = ["binomexp"]


def binomexp(x, n, delta):
    """Binomial expansion of (x + d)^n, term by term with the remainder.

    Parameters
    ----------
    x : float
        Base.
    n : int
        Power, >= 0.
    delta : float
        Shift.

    Returns
    -------
    RichResult
        Keys: terms, sum, exact, abs_error.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (7.35).
    """
    x_f, d_f = float(x), float(delta)
    n_i = int(n)
    if n_i < 0:
        raise ValueError("n must be >= 0")
    terms = [math.comb(n_i, k) * x_f ** (n_i - k) * d_f ** k
             for k in range(n_i + 1)]
    total = float(sum(terms))
    exact = (x_f + d_f) ** n_i
    payload = {
        "terms": terms,
        "sum": total,
        "exact": exact,
        "abs_error": abs(total - exact),
    }
    lines = [("expansion sum", total), ("(x+d)^n", exact)]
    return RichResult(
        title="Binomial expansion of (x + d)^n, term by term with the remainder.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "binomexp: Term-by-term binomial expansion of (x+d)^n. Morin (2016) eq (7.35)."
