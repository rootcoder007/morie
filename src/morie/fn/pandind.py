"""And-rule for independent events: P(A1 and ... and Ak) = prod P(Ai).

Morin (2016), Probability: For the Enthusiastic Beginner, eqs (2.2), (2.3), (2.70).
"""

from . import _array_core as np

from . import _morin
from ._richresult import RichResult

__all__ = ["pandind"]


def pandind(ps):
    """P(A1 and ... and Ak) = prod P(Ai) for independent events.

    Parameters
    ----------
    ps : array-like
        Marginal probabilities, each in [0, 1].

    Returns
    -------
    RichResult
        Keys: ps, p_and; plus p_a and p_b when exactly two events are
        given (kept for the eq (2.2) callers).

    References
    ----------
    Morin, D. J. (2016). Probability: For the Enthusiastic Beginner.
    Createspace Independent Publishing. Eqs. (2.2), (2.3), (2.70).
    """
    value = _morin.prob_and_independent(ps)
    ps_f = [float(x) for x in np.atleast_1d(ps)]
    payload = {"ps": ps_f, "p_and": value}
    if len(ps_f) == 2:
        payload["p_a"] = ps_f[0]
        payload["p_b"] = ps_f[1]
    lines = [("P(all events)", value)]
    return RichResult(
        title="And rule for independent events: P(A and B) = P(A) P(B).",
        summary_lines=lines,
        payload=payload,
    )


def cheatsheet():
    return "pandind: P(A1 and ... and Ak) = prod P(Ai). Morin (2016) eqs (2.2), (2.3), (2.70)."
