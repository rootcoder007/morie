# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Likelihood L(theta) = prod f(X_i; theta)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_likelihood"]


def wasserman_likelihood(data, f, theta):
    """
    Likelihood of theta on i.i.d. data.

    Formula: L(theta) = prod_i f(X_i; theta). Computed through the
    log domain (exp of the summed logs) so long samples do not
    underflow before the product finishes; the log-likelihood ships
    alongside. A zero density at any observation makes L exactly 0
    (log-likelihood -inf), reported rather than raised. ``f = None``
    means the unit-rate exponential density e^{-x/theta}/theta with
    theta > 0.

    Parameters
    ----------
    data : array-like
        Sample (non-empty).
    f : callable or None
        Density f(x, theta) vectorised over x.
    theta : float
        Parameter value.

    Returns
    -------
    result : dict
        Keys: estimate (L), log_likelihood, theta, n, method.

    References
    ----------
    Wasserman (2004), Ch 9, Definition 9.6.

    Examples
    --------
    Exponential(theta = 1) at data summing to 3: L = e^{-3}.

    >>> import math
    >>> out = wasserman_likelihood([1.0, 2.0], None, 1.0)
    >>> abs(out["estimate"] - math.e ** -3) < 1e-15
    True
    >>> round(out["log_likelihood"], 12)
    -3.0
    >>> out2 = wasserman_likelihood([1.0, 2.0], None, 2.0)
    >>> round(out2["log_likelihood"], 12) == round(-2 * math.log(2) - 1.5, 12)
    True
    >>> wasserman_likelihood([-1.0], None, 1.0)["estimate"]
    0.0
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    if data.size == 0:
        raise ValueError("the likelihood on an empty sample is undefined.")
    theta = float(theta)
    if f is None:
        if theta <= 0:
            raise ValueError(f"the exponential model needs theta > 0; got {theta}.")
        f = lambda x, th: np.where(x >= 0, np.exp(-x / th) / th, 0.0)
    dens = np.asarray(f(data, theta), dtype=float)
    if np.any(dens < 0):
        raise ValueError("a density cannot be negative.")
    with np.errstate(divide="ignore"):
        ll = float(np.sum(np.log(dens)))
    L = float(np.exp(ll)) if np.isfinite(ll) else 0.0
    return RichResult(payload={
        "estimate": L, "log_likelihood": ll, "theta": theta,
        "n": int(data.size),
        "method": "L(theta) = prod f(X_i;theta) via log domain"})


def cheatsheet():
    return "wsmlik: L = exp(sum log f); zero density -> L = 0, ll = -inf"
