# morie.fn -- function file (rootcoder007/morie)
"""Cumulative incidence function -- alias of :mod:`morie.fn.crrcim`.

`cumcif` and `crrcim` document the SAME estimator, the Aalen-Johansen
cumulative incidence F_k(t) = integral S(u-) lambda_k(u) du.  A second
implementation would agree with the first at 1e-9 forever and establish
nothing, so this module forwards.
"""

from .crrcim import aalen_johansen, cumulative_incidence

__all__ = ["cumulative_incidence_function"]


def cumulative_incidence_function(time, cause, event_type=None):
    """
    Cumulative incidence function

    Formula: F_k(t) = integral_0^t S(u-) lambda_k(u) du

    Same estimator as :func:`morie.fn.crrcim.cumulative_incidence`.
    Here ``cause`` carries the per-subject event-type vector, matching
    this module's own stub signature; pass ``event_type`` to name the
    cause of interest instead.

    Parameters
    ----------
    time : array-like
        Follow-up time per subject.
    cause : array-like
        0 for censored, otherwise the cause label, one per subject.
    event_type : int or None
        Cause of interest; None uses 1.

    Returns
    -------
    result : dict
        As :func:`morie.fn.crrcim.cumulative_incidence`.

    References
    ----------
    Aalen & Johansen (1978), Scand. J. Statist. 5(3):141-150.
    """
    k = 1 if event_type is None else event_type
    return cumulative_incidence(time, cause, k)


def cheatsheet():
    return "cumcif: cumulative incidence function (alias of crrcim)"


# compact alias per ledger/NAMING.md
cumulativeincidencefunction = cumulative_incidence_function
