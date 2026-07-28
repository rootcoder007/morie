# morie.fn -- function file (rootcoder007/morie)
"""AMSE of sample quantile estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["fauzi_quantile_amse"]


def fauzi_quantile_amse(p, n, f_at_quantile=None, Q_prime=None):
    r"""Asymptotic mean squared error of the sample quantile
    (Fauzi Eq. 3.3):

    .. math:: \mathrm{AMSE}\big(\hat Q(p)\big)
              = \frac{p(1-p)}{n\,f^2(F^{-1}(p))}
              = Q'(p)^2\,\frac{p(1-p)}{n} .

    The two forms are the same quantity written through the density
    or through the quantile function's derivative, since
    :math:`Q'(p) = 1/f(F^{-1}(p))`.

    The formula explains the tail problem in one line. As
    :math:`p \to 0` or 1 the numerator :math:`p(1-p)` shrinks, but
    :math:`f(F^{-1}(p))` shrinks FASTER for any distribution with
    thinning tails, so the AMSE grows. Extreme quantiles are hard
    not because data are scarce in some vague sense but because the
    density there is small, and the module returns both pieces so
    which term dominates is visible.

    Parameters
    ----------
    p : float or array-like
        Probability levels in (0, 1).
    n : int
        Sample size.
    f_at_quantile : array-like, optional
        The density at the quantile.
    Q_prime : array-like, optional
        The quantile function's derivative; the reciprocal route.

    Returns
    -------
    RichResult
        keys: ``p``, ``amse``, ``se``, ``binomial_part``,
        ``density_part``, ``n``, ``tail_note``, ``method``.
    References
    ----------
    Fauzi and Maesono (2023), Eq. (3.3). From the PDF.
    """
    pv = np.atleast_1d(np.asarray(p, dtype=float)).ravel()
    if np.any((pv <= 0) | (pv >= 1)):
        raise ValueError("probability levels must lie strictly in (0, 1).")
    nn = int(n)
    if nn < 2:
        raise ValueError(f"n must be at least 2, got {nn}.")
    if f_at_quantile is None and Q_prime is None:
        raise ValueError("supply either the density at the quantile or Q'(p); "
                         "the AMSE is not determined by p and n alone.")
    if Q_prime is not None:
        qp = np.atleast_1d(np.asarray(Q_prime, dtype=float)).ravel()
        if qp.size != pv.size:
            raise ValueError(f"Q_prime has {qp.size} entries for {pv.size}.")
        dens = 1.0 / np.where(qp != 0, qp, np.nan)
    else:
        dens = np.atleast_1d(np.asarray(f_at_quantile, dtype=float)).ravel()
        if dens.size != pv.size:
            raise ValueError(f"f_at_quantile has {dens.size} for {pv.size}.")
        if np.any(dens <= 0):
            raise ValueError("the density at the quantile must be positive.")
        qp = 1.0 / dens
    binom = pv * (1 - pv) / nn
    amse = qp ** 2 * binom
    return RichResult(payload={
        "p": pv, "amse": amse, "se": np.sqrt(np.maximum(amse, 0.0)),
        "binomial_part": binom, "density_part": qp ** 2, "n": nn,
        "tail_note": "as p goes to 0 or 1 the binomial part shrinks but the "
                     "density part grows faster for thinning tails, so the "
                     "AMSE increases -- that is why extreme quantiles are hard",
        "method": "AMSE of the sample quantile (3.3); p(1-p)/(n f^2) = Q'(p)^2 p(1-p)/n"})


def cheatsheet():
    return "fzamse: extreme quantiles are hard because f(F^{-1}(p)) shrinks faster than p(1-p)"
