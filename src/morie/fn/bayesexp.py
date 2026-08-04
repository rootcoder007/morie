"""Bayes' theorem in explicit form, denominator expanded over A and not-A.

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (2.52), (2.58)-(2.62).
"""

from . import _morin
from ._richresult import RichResult

__all__ = ["bayesexp"]


def bayesexp(p_a=0.02, p_z_given_a=0.95, p_z_given_not_a=0.1):
    """P(A|Z) = P(Z|A)P(A) / [P(Z|A)P(A) + P(Z|~A)P(~A)].

    The defaults are the book's false-positive worked example: a 2%
    prevalence, a 95% sensitive test with a 10% false-positive rate.

    Parameters
    ----------
    p_a : float
        Prior P(A), in [0, 1].
    p_z_given_a : float
        Likelihood P(Z | A), in [0, 1].
    p_z_given_not_a : float
        Likelihood P(Z | not A), in [0, 1].

    Returns
    -------
    RichResult
        Keys: posterior, setup.

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs. (2.52), (2.58)-(2.62).
    """
    value = _morin.bayes_explicit(p_a, p_z_given_a, p_z_given_not_a)
    payload = {
        "posterior": value,
        "setup": {"A": "hypothesis", "not_A": "complement", "Z": "observed evidence"},
    }
    lines = [("P(A|Z)", value)]
    return RichResult(
        title="Bayes' theorem, explicit form over A and not-A.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "bayesexp: P(A|Z) with the denominator expanded over A and not-A. Morin (2016) eq (2.52)."
