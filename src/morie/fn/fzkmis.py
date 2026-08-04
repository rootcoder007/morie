# morie.fn -- function file (rootcoder007/morie)
"""MISE of the standard kernel distribution function estimator."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["kdfmise", "fauzi_kdfe_mise"]


def kdfmise(n, h, rfp, varint, mu2=1.0, r1=None):
    r"""MISE of the standard kernel distribution function estimator.

    The MISE display of Sec. 2.1, assembled from (2.3) and (2.4):

    .. math:: \mathrm{MISE}(\hat F_h) = \frac{h^4}4
              \Big[\!\int\! z^2K(z)dz\Big]^2\!\!\int\![f_X'(x)]^2dx
              + \frac1n\!\int\! F_X(1-F_X)dx - \frac{2h}n r_1
              + o\!\big(h^4 + \tfrac hn\big).

    Differentiating in ``h`` and setting to zero gives

    .. math:: h_{opt} = \Big(\frac{2r_1}{n\mu_2^2R(f_X')}\Big)^{1/3},

    a CUBE root -- the reason this suite's bandwidth rule is
    :math:`n^{-1/3}` and not the density estimator's :math:`n^{-1/5}`.
    For a Gaussian kernel against a normal reference it collapses to
    :math:`4^{1/3}\sigma n^{-1/3}`, which is what
    ``morie.fn._fauzi.kdfe_bandwidth`` returns and what Sec. 5.3.2
    attributes to Azzalini.

    ``hopt`` is returned alongside the MISE so the two never drift apart.
    When ``rfp`` is zero the bias term vanishes and no optimum exists;
    ``hopt`` is then NaN rather than an infinity dressed up as a number.

    Parameters
    ----------
    n : int
        Sample size.
    h : float
        Bandwidth.
    rfp : float
        ``int [f_X'(x)]^2 dx``, the roughness of the density.
    varint : float
        ``int F_X(x)(1 - F_X(x)) dx``.
    mu2 : float, default 1.0
        ``int z^2 K(z) dz``.
    r1 : float, optional
        Kernel constant (2.9); defaults to the Gaussian value.

    Returns
    -------
    RichResult
        Keys ``mise``, ``biasterm``, ``varterm``, ``smoothgain``, ``hopt``, ``r1``, ``method``.

    References
    ----------
    Fauzi and Maesono (2023), Sec. 2.1, Eqs. (2.3)-(2.4) and the MISE display.
    """
    from .fzr1 import kdfr1

    n = int(n)
    h = float(h)
    if n < 1:
        raise ValueError(f"sample size must be at least 1, got {n}.")
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}.")
    if r1 is None:
        r1 = float(kdfr1()["estimate"])
    mu2 = float(mu2)
    biasterm = h ** 4 / 4.0 * mu2 ** 2 * float(rfp)
    varterm = float(varint) / n
    gain = 2.0 * h / n * float(r1)
    if float(rfp) > 0:
        hopt = float((2.0 * float(r1) / (n * mu2 ** 2 * float(rfp))) ** (1.0 / 3.0))
    else:
        hopt = float("nan")
    return RichResult(
        payload={
            "mise": float(biasterm + varterm - gain),
            "biasterm": float(biasterm),
            "varterm": float(varterm),
            "smoothgain": float(gain),
            "hopt": hopt,
            "r1": float(r1),
            "method": "MISE of the standard KDFE (Sec. 2.1)",
        }
    )


fauzi_kdfe_mise = kdfmise


def cheatsheet():
    return "fzkmis: KDFE MISE and its CUBE-root optimal bandwidth (Sec. 2.1)"


# CANONICAL TEST
# >>> r = kdfmise(n=100, h=0.3, rfp=0.2, varint=0.5)
# >>> abs(r['hopt'] - (2 * r['r1'] / (100 * 0.2)) ** (1 / 3)) < 1e-12
# True
