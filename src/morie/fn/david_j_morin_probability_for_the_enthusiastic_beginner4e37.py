"""(1 - lambda eps)^n ~ e^(-n lambda eps) inside the Poisson limit.

Implements eq (4.37) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_37"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_37(lam_eps, n):
    """(1 - lambda eps)^n ~ e^(-n lambda eps) inside the Poisson limit.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.37).
    """
    x = float(lam_eps)
    n_i = int(n)
    if not 0 <= x < 1 or n_i < 1:
        raise ValueError("need 0 <= lam_eps < 1 and n >= 1")
    exact = (1.0 - x) ** n_i
    approx = math.exp(-n_i * x)
    payload = {"exact": exact, "approx": approx,
               "rel_error": abs(exact - approx) / max(exact, 1e-300)}
    lines = [("(1-x)^n", exact), ("e^(-nx)", approx)]
    return RichResult(
        title="(1 - lambda eps)^n ~ e^(-n lambda eps) inside the Poisson limit.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e37: (1 - lambda eps)^n ~ e^(-n lambda eps) inside the Poisson limit. Morin (2016) eq (4.37)."
