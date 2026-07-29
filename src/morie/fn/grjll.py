# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Johnson-Lindenstrauss bound on the random-projection target dimension."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_johnson_lindenstrauss_bound"]

_METHOD = "Johnson-Lindenstrauss minimum dimension"


def geron_johnson_lindenstrauss_bound(n_samples, eps):
    r"""Smallest safe target dimension for a random projection.

    .. math::
        d \ge \frac{4 \log m}{\varepsilon^2/2 - \varepsilon^3/3}

    Two facts about this bound are the reason random projection works
    at all: ``d`` grows only *logarithmically* in the number of
    samples, and it does not depend on the original dimension whatever.
    A million points need barely more room than a thousand.

    The price is ``eps``: the bound blows up like
    :math:`\varepsilon^{-2}`, so halving the allowed distortion
    roughly quadruples the required dimension.

    Parameters
    ----------
    n_samples : int
        Number of points ``m``, at least 2 (``log 1 = 0`` would make the
        bound vacuous).
    eps : float or array-like
        Distortion tolerance in ``(0, 1)``.

    Returns
    -------
    RichResult
        Payload keys ``min_dimension``, ``denominator``, ``eps``,
        ``n_samples``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 7, Random Projection section (JL lemma).

    Examples
    --------
    Ten thousand points at 10% distortion:

    >>> r = geron_johnson_lindenstrauss_bound(10000, 0.1)
    >>> r["min_dimension"]
    7895

    Halving the distortion costs roughly four times the dimension:

    >>> geron_johnson_lindenstrauss_bound(10000, 0.05)["min_dimension"]
    30490

    A hundred times more points barely moves the bound -- the log at
    work:

    >>> geron_johnson_lindenstrauss_bound(1000000, 0.1)["min_dimension"]
    11842
    """
    m = int(n_samples)
    if m < 2:
        raise ValueError(f"n_samples must be at least 2, got {m}; log(1) = 0 makes the bound vacuous.")
    e = np.asarray(eps, dtype=float)
    if not np.all(np.isfinite(e)):
        raise ValueError("eps must be finite.")
    if np.any(e <= 0) or np.any(e >= 1):
        raise ValueError(f"eps must lie strictly in (0, 1), got {e.tolist()}.")

    denom = e**2 / 2.0 - e**3 / 3.0
    d = 4.0 * math.log(m) / denom
    d_int = np.ceil(d).astype(int)
    scalar = e.ndim == 0

    return RichResult(
        title="Johnson-Lindenstrauss bound",
        summary_lines=[("Samples", m), ("eps", e.tolist()),
                       ("Min dimension", d_int.tolist())],
        payload={
            "min_dimension": int(d_int) if scalar else d_int.tolist(),
            "exact": float(d) if scalar else d.tolist(),
            "denominator": float(denom) if scalar else denom.tolist(),
            "eps": float(e) if scalar else e.tolist(),
            "n_samples": m,
            "estimate": int(d_int) if scalar else d_int.tolist(),
            "n": m,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grjll: d >= 4 log(m) / (eps^2/2 - eps^3/3); log in m, independent of input dimension"
