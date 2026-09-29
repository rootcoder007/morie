# morie.fn -- function file (rootcoder007/morie)
"""Bandwidth selection: Silverman's rule of thumb."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["kbwrt"]

# Canonical bandwidths delta_0(K) = (R(K) / mu_2(K)^2)^(1/5) (Marron and
# Nolan 1988); R(K) = int K^2, mu_2(K) = int u^2 K.
_R_MU2 = {
    "gaussian": (1.0 / (2.0 * math.sqrt(math.pi)), 1.0),
    "epanechnikov": (3.0 / 5.0, 1.0 / 5.0),
    "biweight": (5.0 / 7.0, 1.0 / 7.0),
    "triweight": (350.0 / 429.0, 1.0 / 9.0),
    "uniform": (1.0 / 2.0, 1.0 / 3.0),
}


def _canonical_ratio(kernel):
    """delta_0(kernel) / delta_0(gaussian): rescales a Gaussian bandwidth."""
    if kernel not in _R_MU2:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose from {list(_R_MU2)}.")

    def d0(k):
        r, m2 = _R_MU2[k]
        return (r / (m2 * m2)) ** 0.2

    return d0(kernel) / d0("gaussian")


def _q7(xs, p):
    """Type-7 (linear interpolation) sample quantile of sorted ``xs``."""
    h = (len(xs) - 1) * p
    lo = int(math.floor(h))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])


def kbwrt(data, *, kernel: str = "gaussian") -> dict:
    r"""
    Silverman's rule-of-thumb bandwidth selector.

    For the Gaussian kernel, Silverman's rule (3.31):

    .. math::

        h = 0.9 \min\!\left(\hat\sigma,\;\frac{\mathrm{IQR}}{1.34}\right) n^{-1/5}

    (``stats::bw.nrd0`` in R). For another kernel the Gaussian bandwidth
    is multiplied by the ratio of canonical bandwidths
    :math:`\delta_0(K)/\delta_0(\phi)`, :math:`\delta_0(K) =
    (R(K)/\mu_2(K)^2)^{1/5}`, which makes the two kernels smooth
    equivalently (Marron and Nolan 1988): 2.2138 for the Epanechnikov,
    2.6226 for the biweight, 2.9780 for the triweight kernel. (The
    constants 2.34, 2.78 and 3.15 of Silverman's Table 3.1 are the
    normal-reference bandwidths themselves, to be used in place of 1.06,
    not multipliers of the 0.9 rule.)

    Parameters
    ----------
    data : array-like
        1-d array of observations.
    kernel : str
        ``'gaussian'``, ``'epanechnikov'``, ``'biweight'``, ``'triweight'``,
        ``'uniform'``.

    Returns
    -------
    dict
        ``bw``, ``sigma``, ``iqr``, ``n``, ``kernel``, ``ratio``.

    References
    ----------
    Silverman, B. W. (1986). *Density Estimation for Statistics and Data
        Analysis*. Chapman & Hall. Rule (3.31).
    Marron, J. S. and Nolan, D. (1988). Canonical kernels for density
        estimation. *Statistics & Probability Letters*, 7, 195-199.

    Examples
    --------
    >>> round(kbwrt([1.0, 2.0, 4.0, 7.0, 11.0])["bw"], 10)
    2.4339615571
    """
    x = sorted(float(v) for v in (data.tolist() if hasattr(data, "tolist") else data))
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 observations.")
    ratio = _canonical_ratio(kernel)
    m = math.fsum(x) / n
    sigma = math.sqrt(math.fsum((v - m) ** 2 for v in x) / (n - 1))
    iqr = _q7(x, 0.75) - _q7(x, 0.25)
    s = min(sigma, iqr / 1.34) if iqr > 0 else sigma
    s = max(s, 1e-10)
    bw = 0.9 * s * n ** (-0.2) * ratio
    return RichResult(payload={"bw": float(bw), "sigma": sigma, "iqr": iqr, "n": n, "kernel": kernel, "ratio": ratio})


def cheatsheet() -> str:
    return "kbwrt({data}) -> Silverman's rule-of-thumb bandwidth, canonical-kernel rescaling."
