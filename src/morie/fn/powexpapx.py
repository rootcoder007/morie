"""First-order approximation (1 + a)^n ~ e^(na), valid when n a^2 << 1.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (7.14), (7.23).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["powexpapx"]


def powexpapx(a, n):
    """First-order approximation (1 + a)^n ~ e^(na), valid when n a^2 << 1.

    ``valid`` is the decision n a^2 < 0.1, the book's rule of thumb for
    when the first-order form may be used.

    Parameters
    ----------
    a : float
        Perturbation, > -1.
    n : float
        Exponent.

    Returns
    -------
    RichResult
        Keys: exact, approx, na2, valid.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs (7.14), (7.23).
    """
    exact, approx, validity = _morin.one_plus_a_to_n(a, n, order=1)
    payload = {
        "exact": exact,
        "approx": approx,
        "na2": validity,
        "valid": validity < 0.1,
    }
    lines = [("(1+a)^n", exact), ("e^(na)", approx), ("na^2", validity)]
    return RichResult(
        title="First-order approximation (1 + a)^n ~ e^(na), valid when n a^2 << 1.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "powexpapx: (1+a)^n ~ e^(na) to first order. Morin (2016) eqs (7.14), (7.23)."
