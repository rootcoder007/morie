# morie.fn -- function file (rootcoder007/morie)
"""Effective simple-random-sample size of a weighted design."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["neffsrs", "effective_srs"]


def neffsrs(w):
    """Kish effective sample size and design effect of unequal weights.

    Kish's measure of the loss of precision caused by unequal selection
    weights writes the design effect as one plus the squared coefficient
    of variation of the weights,

        deff = 1 + cv^2(w) = n * sum w_i^2 / (sum w_i)^2,

    so that the complex design carries as much information as a simple
    random sample of

        n_eff = n / deff = (sum w_i)^2 / sum w_i^2

    observations.

    Parameters
    ----------
    w : array-like
        Positive selection or post-stratification weights, one per unit.

    Returns
    -------
    RichResult
        ``neff``, ``deff``, ``cv2``, ``n``, ``sumw``, ``sumw2``.

    References
    ----------
    Kish, L. (1965), Survey Sampling, Wiley, Sect. 11.7 ("the effect of
    unequal weights on the variance"), where deff = 1 + cv^2(w) and the
    effective sample size is n divided by it.  Standard published form;
    the 1965 monograph was not available in the local corpus and was not
    read for this implementation.
    """
    w = C.vec(w)
    n = len(w)
    if n == 0:
        raise ValueError("w must be non-empty")
    if any(v <= 0.0 for v in w):
        raise ValueError("weights must be strictly positive")
    s1 = sum(w)
    s2 = sum(v * v for v in w)
    deff = n * s2 / (s1 * s1)
    return RichResult(payload={
        "neff": s1 * s1 / s2, "deff": deff, "cv2": deff - 1.0,
        "n": n, "sumw": s1, "sumw2": s2,
        "method": "Kish effective sample size, deff = 1 + cv^2(w)"})


effective_srs = neffsrs


effectivesrs = neffsrs


def cheatsheet():
    return "adjsrs: Effective simple-random-sample size of a weighted design."
