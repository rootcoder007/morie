# morie.fn -- function file (rootcoder007/morie)
"""Manski no-assumption bounds on a partially observed mean (forwards to bndest)."""

from .bndest import bound_estimation

__all__ = ["manskif"]


def manskif(y, observed, support, treatment=None):
    """Manski (1989, 1990) worst-case bounds -- forwarder.

    ``manskif`` and ``bndest`` (and the R arm ``morie_bnd_manski``)
    document the SAME method: the mean of a partially observed outcome
    is bounded by filling the unseen part with the support endpoints,

        E[Y] in [ E[Y|obs] P(obs) + K0 (1-P(obs)),
                  E[Y|obs] P(obs) + K1 (1-P(obs)) ],

    and with ``treatment`` given the same construction bounds each
    potential-outcome mean and the ATE.  This function forwards to
    :func:`morie.fn.bndest.bound_estimation`; a second implementation
    would agree with the first at 1e-9 forever while doubling the
    surface.  (The stub-era name ``manski_bounds`` is NOT re-exported
    here: that name belongs to ``morie.fn.bnd``.)

    Parameters
    ----------
    y : array-like
        Outcome; entries where ``observed`` is False are ignored.
    observed : array-like of bool, or None
        Whether the outcome was seen (ignored when ``treatment`` given).
    support : (float, float)
        Logical range ``[K0, K1]`` of the outcome.
    treatment : array-like of 0/1, optional
        Treatment indicator for the ATE version.

    Returns
    -------
    RichResult
        As :func:`morie.fn.bndest.bound_estimation`.

    References
    ----------
    Manski, C. F. (1989), "Anatomy of the Selection Problem", Journal of
    Human Resources 24(3):343-360.  Manski, C. F. (1990), "Nonparametric
    Bounds on Treatment Effects", AER P&P 80(2):319-323.  Molinari, F.
    (2021), Handbook of Econometrics 7A, eq. (2.11); local source
    ~/work/scratch/x000/molinari.pdf (arXiv:2004.11751).
    """
    return bound_estimation(y, observed, support, treatment=treatment)


def cheatsheet():
    return "manskif: Manski worst-case bounds -- forwards to bndest"
