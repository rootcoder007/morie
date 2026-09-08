# morie.fn -- internal helpers (rootcoder007/morie)
"""Design-based survey estimation.

The organising fact is that in a design-based analysis the
POPULATION values are fixed and the randomness is the SAMPLE
INCLUSION. Variance formulas therefore come from the inclusion
probabilities, not from a model for y, which is why every estimator
here takes ``pi`` or weights rather than a distributional assumption.
"""

from . import _array_core as np

__all__ = ["check_weights", "ht_total", "hajek_mean", "linearise",
           "srs_variance"]


def check_weights(w, n, name="weights"):
    """Validate and return design weights of the right length."""
    wv = np.atleast_1d(np.asarray(w, dtype=float)).ravel()
    if wv.size != n:
        raise ValueError(f"{name} has {wv.size} entries for {n} observations.")
    if np.any(wv < 0):
        raise ValueError(f"{name} must be non-negative.")
    if not np.any(wv > 0):
        raise ValueError(f"{name} are all zero.")
    return wv


def ht_total(y, pi):
    r"""Horvitz-Thompson total :math:`\sum y_i/\pi_i`.

    Unbiased for ANY design with strictly positive inclusion
    probabilities, which is its whole appeal -- no model, no
    assumption about y. The price is that it is not
    scale-equivariant: it does not use the fact that the weights
    should sum to the population size, so it can be wildly variable
    when the weights are.
    """
    yv = np.asarray(y, dtype=float).ravel()
    p = np.asarray(pi, dtype=float).ravel()
    if p.size != yv.size:
        raise ValueError(f"pi has {p.size} entries for {yv.size} observations.")
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError("inclusion probabilities must lie in (0, 1].")
    return float(np.sum(yv / p))


def hajek_mean(y, pi):
    r"""Hajek ratio estimator
    :math:`\sum(y_i/\pi_i)/\sum(1/\pi_i)`.

    Ratio-adjusts the Horvitz-Thompson total by the estimated
    population size. Biased in finite samples -- it is a ratio of two
    random quantities -- but usually far less variable, because
    errors in the numerator and denominator move together. The
    trade is bias for variance, and it is the reason survey packages
    default to it.
    """
    yv = np.asarray(y, dtype=float).ravel()
    p = np.asarray(pi, dtype=float).ravel()
    if p.size != yv.size:
        raise ValueError(f"pi has {p.size} entries for {yv.size} observations.")
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError("inclusion probabilities must lie in (0, 1].")
    w = 1.0 / p
    return float(np.sum(w * yv) / np.sum(w))


def linearise(grad, cov):
    r"""Delta-method variance
    :math:`(\nabla g)'\Sigma(\nabla g)`."""
    g = np.atleast_1d(np.asarray(grad, dtype=float)).ravel()
    S = np.atleast_2d(np.asarray(cov, dtype=float))
    if S.shape != (g.size, g.size):
        raise ValueError(f"cov must be {g.size} by {g.size}, got {S.shape}.")
    return float(g @ S @ g)


def srs_variance(y, n, N=None):
    r"""Simple-random-sample variance of a mean with the finite
    population correction :math:`(1 - n/N)S^2/n`.

    The correction is not decoration: at a 10 percent sampling
    fraction it removes a tenth of the variance, and at a census it
    correctly gives zero. Omitting it -- treating a survey sample as
    i.i.d. draws from an infinite population -- overstates the
    standard error, sometimes badly.
    """
    yv = np.asarray(y, dtype=float).ravel()
    if yv.size < 2:
        raise ValueError(f"need at least 2 observations, got {yv.size}.")
    s2 = float(np.var(yv, ddof=1))
    fpc = 1.0 if N is None else max(0.0, 1.0 - float(n) / float(N))
    return fpc * s2 / float(n), fpc


def cheatsheet():
    return "_survey: randomness is the SAMPLE, not y -- variance comes from pi, plus the fpc"
