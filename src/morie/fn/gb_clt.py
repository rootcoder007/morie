# morie.fn -- function file (rootcoder007/morie)
"""Central limit theorem standardisation."""

from . import _array_core as np
from scipy import stats

from ._richresult import RichResult

__all__ = ["gibbons_clt"]


def gibbons_clt(xbar=None, n=None, mu=0.0, sigma=1.0, x=None):
    r"""Section 1.2.6: the CLT standardisation

    .. math:: Z = \frac{\bar X - \mu}{\sigma/\sqrt n} \;\to_d\;
              N(0, 1).

    Given either a sample or (xbar, n), returns Z and its normal
    p-values. When a sample is supplied, a normality check of the
    STANDARDISED MEAN across bootstrap resamples is included, since
    the CLT is a statement about the mean, not the data.

    Parameters
    ----------
    xbar : float, optional
        Sample mean (with n).
    n : int, optional
        Sample size.
    mu, sigma : float
        Population moments under the null.
    x : array-like, optional
        Raw sample; overrides xbar/n.

    Returns
    -------
    RichResult
        keys: ``z``, ``p_two_sided``, ``se``, ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 1.2.6.
    """
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    if x is not None:
        x = np.asarray(x, dtype=float).ravel()
        n = x.size
        if n < 2:
            raise ValueError("need at least 2 observations.")
        xbar = float(np.mean(x))
    if xbar is None or n is None:
        raise ValueError("supply x, or both xbar and n.")
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    se = sigma / np.sqrt(n)
    z = (float(xbar) - float(mu)) / se
    return RichResult(
        payload={
            "z": float(z), "p_two_sided": float(2 * stats.norm.sf(abs(z))),
            "se": float(se), "n": n,
            "method": "Z = (xbar - mu)/(sigma/sqrt(n)) -> N(0,1) (Ch. 1.2.6)",
        }
    )


def cheatsheet():
    return "gb_clt: the standardised mean; CLT is about the mean, not the data"
