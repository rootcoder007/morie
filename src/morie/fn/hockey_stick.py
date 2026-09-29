"""Hockey-stick identity: sum_{j=k-1}^{n-1} C(j,k-1) = C(n,k).

Implements eq (1.29) of Morin (2016), Probability: For the
Enthusiastic Beginner.
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["hockey_stick"]


def hockey_stick(n, k):
    """Hockey-stick identity: sum_{j=k-1}^{n-1} C(j,k-1) = C(n,k).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (1.29).

    Examples
    --------
    >>> hockey_stick(7, 3)["sum"]
    35
    """
    lhs, rhs = _morin.hockey_stick(n, k)
    payload = {"n": int(n), "k": int(k), "sum": lhs, "binomial": rhs, "identity_holds": lhs == rhs}
    lines = [("sum along diagonal", lhs), ("C(n, k)", rhs)]
    return RichResult(
        title="Hockey-stick identity: sum_{j=k-1}^{n-1} C(j,k-1) = C(n,k).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner1e29: Hockey-stick identity: sum_{j=k-1}^{n-1} C(j,k-1) = C(n,k). Morin (2016) eq (1.29)."


# compact alias per ledger/NAMING.md
hockeystick = hockey_stick
