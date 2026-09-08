"""Full product expansion (1+a)^n = e^(na) e^(-na^2/2) e^(na^3/3) ...

Morin (2016), Probability: For the Enthusiastic Beginner, eq (7.21).
"""

import math

from ._richresult import RichResult

__all__ = ["powlogser"]


def powlogser(a, n, terms=12):
    """Full product expansion (1+a)^n = e^(na) e^(-na^2/2) e^(na^3/3) ...

    Parameters
    ----------
    a : float
        Perturbation with |a| < 1, so the log series converges.
    n : float
        Exponent.
    terms : int
        Number of log-series terms, >= 1.

    Returns
    -------
    RichResult
        Keys: exact, product_form, rel_error.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eq (7.21).
    """
    a_f, n_f = float(a), float(n)
    if a_f <= -1.0 or abs(a_f) >= 1.0:
        raise ValueError("need |a| < 1 for the log series")
    s = 0.0
    for j in range(1, int(terms) + 1):
        s += (-1) ** (j + 1) * a_f ** j / j
    product_form = math.exp(n_f * s)
    exact = (1.0 + a_f) ** n_f
    payload = {
        "exact": exact,
        "product_form": product_form,
        "rel_error": abs(exact - product_form) / max(abs(exact), 1e-300),
    }
    lines = [("exact", exact), ("product form", product_form)]
    return RichResult(
        title="Full product expansion (1+a)^n = e^(na) e^(-na^2/2) e^(na^3/3) ...",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "powlogser: (1+a)^n as the exponential of its log series. Morin (2016) eq (7.21)."
