# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""Bandwidth selection via Silverman rule-of-thumb."""

from __future__ import annotations

import math


def bwrot(
    x,
    *,
    kernel: str = "gaussian",
    method: str = "silverman",
) -> dict:
    r"""
    Bandwidth selection via rule-of-thumb (Silverman / Scott).

    **Silverman (1986)** normal reference rule:

    .. math::

        h = 1.06 \cdot \min\!\left(\hat\sigma,\,
        \frac{\text{IQR}}{1.34}\right) n^{-1/5}

    **Scott (1992)** variant:

    .. math::

        h = 1.059 \cdot \hat\sigma \cdot n^{-1/5}

    Parameters
    ----------
    x : np.ndarray
        1-d array of observed values (n,).
    kernel : str
        Kernel: ``'gaussian'``, ``'epanechnikov'``, ``'biweight'``,
        ``'triweight'`` or ``'uniform'``. The Gaussian bandwidth is
        multiplied by the ratio of canonical bandwidths
        :math:`\\delta_0(K)/\\delta_0(\\phi)` (Marron and Nolan 1988):
        2.2138 Epanechnikov, 1.7400 uniform.
    method : str
        ``'silverman'`` or ``'scott'``.

    Returns
    -------
    dict
        Keys: ``bandwidth``, ``sigma``, ``iqr``, ``method``, ``n_obs``,
        ``kernel_ratio``.

    References
    ----------
    Silverman, B. W. (1986). Density Estimation for Statistics and Data
        Analysis. Chapman & Hall.
    Scott, D. W. (1992). Multivariate Density Estimation. Wiley.
    Horowitz, J. L. (2009). Semiparametric and Nonparametric Methods in
        Econometrics. Springer. Chapter 2.
    Marron, J. S. and Nolan, D. (1988). Canonical kernels for density
        estimation. *Statistics & Probability Letters*, 7, 195-199.

    Examples
    --------
    >>> round(bwrot([1.0, 2.0, 4.0, 7.0, 11.0])["bandwidth"], 10)
    2.8666658339
    """
    from .kbwrt import _canonical_ratio, _q7

    xs = sorted(float(v) for v in (x.tolist() if hasattr(x, "tolist") else x))
    n = len(xs)
    if n < 2:
        raise ValueError("Need at least 2 observations.")
    valid_methods = {"silverman", "scott"}
    if method not in valid_methods:
        raise ValueError(f"Unknown method '{method}'. Choose from {valid_methods}.")
    ratio = _canonical_ratio(kernel)
    m = math.fsum(xs) / n
    sigma = math.sqrt(math.fsum((v - m) ** 2 for v in xs) / (n - 1))
    iqr = _q7(xs, 0.75) - _q7(xs, 0.25)
    if method == "silverman":
        spread = min(sigma, iqr / 1.34) if iqr > 0 else sigma
        h = 1.06 * spread * n ** (-1 / 5)
    else:
        h = 1.059 * sigma * n ** (-1 / 5)
    h *= ratio
    return {
        "bandwidth": h,
        "sigma": sigma,
        "iqr": iqr,
        "method": method,
        "n_obs": n,
        "kernel_ratio": ratio,
    }


bwrot_fn = bwrot


def cheatsheet() -> str:
    return "bwrot({x}) -> Bandwidth selection via Silverman/Scott rule-of-thumb."
