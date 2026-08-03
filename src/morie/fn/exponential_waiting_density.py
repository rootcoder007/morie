"""Exponential waiting-time density rho(t) = lambda e^(-lambda t).

Implements eq (4.26) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

import math

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["exponential_waiting_density"]


def exponential_waiting_density(t, lam):
    """Exponential waiting-time density rho(t) = lambda e^(-lambda t).

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (4.26).
    """
    value = _morin.exponential_waiting_density(t, lam)
    payload = {"t": float(t), "lambda": float(lam), "density": value}
    lines = [("rho(t)", value)]
    return RichResult(
        title="Exponential waiting-time density rho(t) = lambda e^(-lambda t).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner4e26: Exponential waiting-time density rho(t) = lambda e^(-lambda t). Morin (2016) eq (4.26)."
