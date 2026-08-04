"""(1 - x)^n against e^(-n x) inside the Poisson limit.

Morin (2016), Probability: For the Enthusiastic Beginner, eq (4.37).
"""

import math

from ._richresult import RichResult

__all__ = ["onemexp"]


def onemexp(lam_eps, n):
    """(1 - x)^n against e^(-n x) inside the Poisson limit.

    Parameters
    ----------
    lam_eps : float
        The per-slot probability lambda eps, in [0, 1).
    n : int
        Number of slots, >= 1.

    Returns
    -------
    RichResult
        Keys: exact, approx, rel_error.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (4.37).
    """
    x = float(lam_eps)
    n_i = int(n)
    if not 0 <= x < 1 or n_i < 1:
        raise ValueError("need 0 <= lam_eps < 1 and n >= 1")
    exact = (1.0 - x) ** n_i
    approx = math.exp(-n_i * x)
    payload = {
        "exact": exact,
        "approx": approx,
        "rel_error": abs(exact - approx) / max(exact, 1e-300),
    }
    lines = [("(1-x)^n", exact), ("e^(-nx)", approx)]
    return RichResult(
        title="(1 - x)^n against e^(-n x) inside the Poisson limit.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "onemexp: (1-x)^n ~ e^(-nx), the Poisson-limit step. Morin (2016) eq (4.37)."
