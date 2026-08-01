"""Full product expansion (1+a)^n = e^(na) e^(-na^2/2) e^(na^3/3) ...

Implements eq (7.21) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

import numpy as np

from . import _morin
from ._richresult import RichResult

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_21"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_7_equation_21(a, n, terms=12):
    """Full product expansion (1+a)^n = e^(na) e^(-na^2/2) e^(na^3/3) ...

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (7.21).
    """
    a_f, n_f = float(a), float(n)
    if a_f <= -1.0 or abs(a_f) >= 1.0:
        raise ValueError("need |a| < 1 for the log series")
    s = 0.0
    for j in range(1, int(terms) + 1):
        s += ((-1) ** (j + 1)) * a_f ** j / j
    product_form = math.exp(n_f * s)
    exact = (1.0 + a_f) ** n_f
    payload = {"exact": exact, "product_form": product_form,
               "rel_error": abs(exact - product_form) / max(abs(exact), 1e-300)}
    lines = [("exact", exact), ("product form", product_form)]
    return RichResult(
        title="Full product expansion (1+a)^n = e^(na) e^(-na^2/2) e^(na^3/3) ...",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner7e21: Full product expansion (1+a)^n = e^(na) e^(-na^2/2) e^(na^3/3) ... Morin (2016) eq (7.21)."
