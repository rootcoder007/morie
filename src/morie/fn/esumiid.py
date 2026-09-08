"""E of a sum of n i.i.d. variables: n E(X).

Morin (2016), Probability: For the Enthusiastic Beginner, eq (3.15).
"""


from ._richresult import RichResult

__all__ = ["esumiid"]


def esumiid(e_x, n):
    """E of a sum of n i.i.d. variables: n E(X).

    Parameters
    ----------
    e_x : float
        The common expectation E(X).
    n : int
        Number of i.i.d. terms, >= 0.

    Returns
    -------
    RichResult
        Keys: e_sum, n.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (3.15).
    """
    n_i = int(n)
    if n_i < 0 or n_i != float(n):
        raise ValueError("n must be a non-negative integer")
    value = n_i * float(e_x)
    payload = {"e_sum": value, "n": n_i}
    lines = [("E(X1 + ... + Xn)", value)]
    return RichResult(
        title="E of a sum of n i.i.d. variables: n E(X).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "esumiid: E(X1 + ... + Xn) = n E(X). Morin (2016) eq (3.15)."
