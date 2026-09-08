# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kullback-Leibler divergence."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_kullback_leibler"]


def wasserman_kullback_leibler(p, q, x_grid=None):
    """
    KL divergence D(p || q), in nats.

    Formula: D = int p log(p/q) dx on a grid, or sum p_i log(p_i/q_i)
    for probability vectors. Convention: terms with p_i = 0
    contribute 0; any point with p_i > 0 and q_i = 0 makes D = inf
    (reported, not raised). D >= 0 always; asymmetry is the point.

    Parameters
    ----------
    p, q : array-like
        Probability vectors (each summing to 1 within 1e-8), or
        densities on x_grid.
    x_grid : array-like, optional
        Support grid; presence selects the continuous form.

    Returns
    -------
    result : dict
        Keys: estimate (nats), bits, reverse (D(q||p)), form, n,
        method.

    References
    ----------
    Wasserman (2004), Ch 23; Kullback & Leibler (1951).

    Examples
    --------
    >>> import math
    >>> out = wasserman_kullback_leibler([0.5, 0.5], [0.25, 0.75])
    >>> abs(out["estimate"] - (0.5 * math.log(2) + 0.5 * math.log(2/3))) < 1e-15
    True
    >>> out["estimate"] >= 0
    True
    >>> wasserman_kullback_leibler([0.5, 0.5], [1.0, 0.0])["estimate"]
    inf
    >>> wasserman_kullback_leibler([1.0, 0.0], [0.5, 0.5])["estimate"] == math.log(2)
    True
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    q = np.atleast_1d(np.asarray(q, dtype=float))
    if p.size != q.size:
        raise ValueError(f"p ({p.size}) and q ({q.size}) lengths differ.")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("probabilities/densities cannot be negative.")

    def _kl(a, b):
        if np.any((a > 0) & (b == 0)):
            return float("inf")
        mask = a > 0
        terms = a[mask] * np.log(a[mask] / b[mask])
        if x_grid is None:
            return float(np.sum(terms))
        x = np.atleast_1d(np.asarray(x_grid, dtype=float))
        full = np.zeros_like(a)
        full[mask] = a[mask] * np.log(a[mask] / b[mask])
        dx = np.diff(x)
        return float(0.5 * np.sum(dx * (full[1:] + full[:-1])))

    if x_grid is None:
        for name, v in (("p", p), ("q", q)):
            s = float(np.sum(v))
            if abs(s - 1.0) > 1e-8:
                raise ValueError(f"{name} must sum to 1; got {round(s, 12)}.")
        form = "discrete"
        n = int(p.size)
    else:
        form = "continuous"
        n = int(np.atleast_1d(np.asarray(x_grid)).size)
    D = _kl(p, q)
    Drev = _kl(q, p)
    return RichResult(payload={
        "estimate": D,
        "bits": float(D / np.log(2.0)) if np.isfinite(D) else float("inf"),
        "reverse": Drev, "form": form, "n": n,
        "method": "KL D(p||q); 0 log 0 = 0, p>0 & q=0 -> inf"})


def cheatsheet():
    return "wsmkbk: D(p||q) = sum p log(p/q); reverse direction alongside"
