# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Johnson-Lindenstrauss lemma: d' = O(log(n)/eps^2) preserves pairwise distances."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_johnson_lindenstrauss"]

_METHOD = "Johnson-Lindenstrauss minimum dimension"


def geron_johnson_lindenstrauss(n, eps):
    """
    Johnson-Lindenstrauss lemma: d' = O(log(n)/eps^2) preserves pairwise distances.

    Formula: d_min >= 4*log(n) / (eps^2/2 - eps^3/3)

    The striking part of the bound is what is *absent* from it: the
    original dimensionality.  ``d_min`` depends only on the number of
    points and the tolerated distortion, so a million-dimensional
    dataset and a thousand-dimensional one with the same ``n`` need the
    same target dimension.  ``log`` is natural throughout.

    The denominator ``eps^2/2 - eps^3/3`` is positive only for
    ``0 < eps < 1``; outside that range the bound is meaningless and
    this raises rather than returning a negative dimension.

    Parameters
    ----------
    n : int
        Number of points, at least 2 (with one point there is nothing to
        distort, and ``log(1) = 0`` would give ``d_min = 0``).
    eps : float or array-like
        Tolerated relative distortion of squared distances, ``0 < eps < 1``.

    Returns
    -------
    result : RichResult
        Keys: d_min, d_min_exact, eps, n_points, estimate, n, method.

    Examples
    --------
    Hand-checkable: with ``n = 2``, ``eps = 0.5`` the denominator is
    ``0.125 - 0.0416666... = 0.0833333...`` and the numerator is
    ``4*log(2) = 2.7725887...``, giving 33.27 -> 34:

    >>> r = geron_johnson_lindenstrauss(2, 0.5)
    >>> round(r["d_min_exact"], 4)
    33.2711
    >>> r["d_min"]
    34

    Halving eps roughly quadruples the requirement:

    >>> a = geron_johnson_lindenstrauss(1000, 0.2)["d_min"]
    >>> b = geron_johnson_lindenstrauss(1000, 0.1)["d_min"]
    >>> a, b
    (1595, 5921)

    Several tolerances at once:

    >>> [int(v) for v in geron_johnson_lindenstrauss(10000, [0.1, 0.5])["d_min"]]
    [7895, 443]

    References
    ----------
    Géron Ch 7
    """
    n_points = int(n)
    if n_points < 2:
        raise ValueError(f"geron_johnson_lindenstrauss: n must be at least 2 points, got {n!r}")
    e = np.asarray(eps, dtype=float)
    scalar = e.ndim == 0
    e = np.atleast_1d(e)
    if e.size == 0:
        raise ValueError("geron_johnson_lindenstrauss: eps is empty")
    if not np.all(np.isfinite(e)):
        raise ValueError("geron_johnson_lindenstrauss: eps contains non-finite values")
    if np.any(e <= 0) or np.any(e >= 1):
        raise ValueError(
            f"geron_johnson_lindenstrauss: eps must lie strictly in (0, 1); the bound's denominator "
            f"eps^2/2 - eps^3/3 is non-positive otherwise (got {e.tolist()})"
        )

    denom = e**2 / 2.0 - e**3 / 3.0
    exact = 4.0 * np.log(n_points) / denom
    d_min = np.ceil(exact).astype(np.int64)

    return RichResult(
        title="Johnson-Lindenstrauss bound",
        summary_lines=[
            ("Points", n_points),
            ("eps", float(e[0]) if scalar else e.tolist()),
            ("Minimum dimension", int(d_min[0]) if scalar else d_min.tolist()),
        ],
        interpretation=(
            "The bound is independent of the original dimensionality; it is also loose, "
            "so a smaller projection often works in practice."
        ),
        payload={
            "d_min": int(d_min[0]) if scalar else d_min,
            "d_min_exact": float(exact[0]) if scalar else exact,
            "eps": float(e[0]) if scalar else e,
            "n_points": n_points,
            "estimate": float(d_min[0]) if scalar else float(np.max(d_min)),
            "n": n_points,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmjl: JL bound d_min = ceil(4 ln n / (eps^2/2 - eps^3/3)); independent of input dimension"
