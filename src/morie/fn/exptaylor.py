"""Taylor series e^x = sum x^k / k!.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (7.7).
"""

from . import _morin

from ._richresult import RichResult

__all__ = ["exptaylor"]


def exptaylor(x, terms=30):
    """Taylor series e^x = sum x^k / k!.

    Parameters
    ----------
    x : float
        Series argument.
    terms : int
        Number of terms, >= 1.

    Returns
    -------
    RichResult
        Keys: partial_sums, e_x, final_error.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (7.7).
    """
    partials, closed = _morin.exp_taylor(x, terms)
    payload = {
        "partial_sums": partials,
        "e_x": closed,
        "final_error": abs(partials[-1] - closed),
    }
    lines = [("series", partials[-1]), ("e^x", closed)]
    return RichResult(
        title="Taylor series e^x = sum x^k / k!.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "exptaylor: e^x = sum x^k / k!. Morin (2016) eq (7.7)."
