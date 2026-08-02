# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Differential entropy H(X) = -int p log p."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_entropy"]


def wasserman_entropy(p, x_grid=None):
    """
    Entropy of a distribution, in nats.

    Formula: H(X) = -int p(x) log p(x) dx for a density on a grid
    (``x_grid`` given), or H = -sum p_i log p_i for a probability
    vector (``x_grid`` omitted). 0 log 0 = 0 by convention. A
    probability vector must sum to 1 within 1e-8.

    Parameters
    ----------
    p : array-like
        Probability vector, or density values on x_grid.
    x_grid : array-like, optional
        Support grid; presence selects the differential form.

    Returns
    -------
    result : dict
        Keys: estimate (nats), bits, form, n, method.

    References
    ----------
    Wasserman (2004), Ch 23 (information theory).

    Examples
    --------
    >>> import math
    >>> out = wasserman_entropy([0.5, 0.5])
    >>> abs(out["estimate"] - math.log(2)) < 1e-15
    True
    >>> out["bits"]
    1.0
    >>> wasserman_entropy([1.0, 0.0])["estimate"]
    0.0
    >>> import numpy as np
    >>> g = np.linspace(0.0, 1.0, 100001)
    >>> abs(wasserman_entropy(np.ones_like(g), g)["estimate"] - 0.0) < 1e-12
    True
    >>> wasserman_entropy([0.4, 0.4])
    Traceback (most recent call last):
        ...
    ValueError: a probability vector must sum to 1; got 0.8.
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    if np.any(p < 0):
        raise ValueError("probabilities/densities cannot be negative.")
    if x_grid is None:
        s = float(np.sum(p))
        if abs(s - 1.0) > 1e-8:
            raise ValueError(f"a probability vector must sum to 1; got {round(s, 12)}.")
        nz = p[p > 0]
        H = float(-np.sum(nz * np.log(nz))) + 0.0
        form = "discrete"
        n = int(p.size)
    else:
        x = np.atleast_1d(np.asarray(x_grid, dtype=float))
        if x.size != p.size or x.size < 2:
            raise ValueError("density and grid must match with >= 2 points.")
        integ = np.where(p > 0, -p * np.log(np.where(p > 0, p, 1.0)), 0.0)
        dx = np.diff(x)
        H = float(0.5 * np.sum(dx * (integ[1:] + integ[:-1])))
        form = "differential"
        n = int(x.size)
    return RichResult(payload={
        "estimate": H, "bits": float(H / np.log(2.0)), "form": form,
        "n": n, "method": f"{form} entropy, nats, 0 log 0 = 0"})


def cheatsheet():
    return "wsment: H = -sum p log p (discrete) or -int p log p (grid); nats + bits"
