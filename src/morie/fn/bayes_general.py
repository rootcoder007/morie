"""Bayes' theorem, general form over a complete hypothesis set.

Implements eq (2.74) of Morin (2016), Probability: For the
Enthusiastic Beginner. The auto-extracted placeholder returned the
sample mean of an arbitrary vector; this module now computes the
book's actual result.
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["bayes_general"]


def bayes_general(priors, likelihoods):
    """Bayes' theorem, general form over a complete hypothesis set.

    Reference
    ---------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner. Createspace Independent Publishing. Eq. (2.74).
    """
    post, p_z = _morin.bayes_general(priors, likelihoods)
    payload = {"posteriors": [float(x) for x in post], "p_z": p_z}
    lines = [("P(Z)", p_z)] + [(f"P(A{i}|Z)", float(p))
                               for i, p in enumerate(post, 1)]
    return RichResult(
        title="Bayes' theorem, general form over a complete hypothesis set.",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2e74: Bayes' theorem, general form over a complete hypothesis set. Morin (2016) eq (2.74)."


# compact alias per ledger/NAMING.md
bayesgeneral = bayes_general
