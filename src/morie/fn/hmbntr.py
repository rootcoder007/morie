# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batch normalization: normalize per-batch then affine rescale."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_batch_normalization"]


def geron_batch_normalization(x, gamma=1.0, beta=0.0, eps=1e-5, momentum=0.9, running_mean=None, running_var=None):
    """
    Batch normalization: normalize per-batch then affine rescale.

    Formula: x_hat = (x - mu_B) / sqrt(var_B + eps); y = gamma * x_hat + beta

    Parameters
    ----------
    x : array-like, shape (n, d) or (n,)
        Mini-batch; statistics are taken per feature column.
    gamma, beta : float or array-like of length d
        Affine scale and shift.
    eps : float
        Variance floor; must be positive.
    momentum : float
        Exponential-moving-average factor for the inference statistics.
    running_mean, running_var : array-like of length d, optional
        Previous inference statistics to update.

    Returns
    -------
    result : RichResult
        Keys: y, x_hat, mu, var, running_mean, running_var, estimate, n, method.

    Examples
    --------
    >>> r = geron_batch_normalization([[1.0], [3.0]], eps=0.0)
    >>> [float(v) for v in r["x_hat"].ravel()]
    [-1.0, 1.0]
    >>> float(r["mu"][0]), float(r["var"][0])
    (2.0, 1.0)
    >>> r2 = geron_batch_normalization([[1.0], [3.0]], gamma=2.0, beta=5.0, eps=0.0)
    >>> [float(v) for v in r2["y"].ravel()]
    [3.0, 7.0]

    References
    ----------
    Géron Ch 11
    """
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"geron_batch_normalization: x must be 1-D or 2-D, got ndim={X.ndim}")
    n, d = X.shape
    if n == 0:
        raise ValueError("geron_batch_normalization: mini-batch is empty")
    if not np.all(np.isfinite(X)):
        raise ValueError("geron_batch_normalization: x must be finite")
    e = float(eps)
    if e < 0:
        raise ValueError("geron_batch_normalization: eps must be non-negative")
    g = np.broadcast_to(np.asarray(gamma, dtype=float).ravel(), (d,)) if np.size(gamma) in (1, d) else None
    b = np.broadcast_to(np.asarray(beta, dtype=float).ravel(), (d,)) if np.size(beta) in (1, d) else None
    if g is None or b is None:
        raise ValueError(f"geron_batch_normalization: gamma and beta must be scalars or length-{d} vectors")

    mu = X.mean(axis=0)
    var = X.var(axis=0, ddof=0)
    denom = np.sqrt(var + e)
    if np.any(denom == 0):
        raise ValueError(
            "geron_batch_normalization: a feature has zero variance and eps=0, so the scale is undefined; "
            "pass eps > 0"
        )
    x_hat = (X - mu) / denom
    y = g * x_hat + b

    mom = float(momentum)
    if not (0.0 <= mom <= 1.0):
        raise ValueError("geron_batch_normalization: momentum must lie in [0, 1]")
    rm = mu if running_mean is None else mom * np.asarray(running_mean, dtype=float).ravel() + (1 - mom) * mu
    # Inference variance uses the unbiased batch estimate, per the original paper.
    var_unb = X.var(axis=0, ddof=1) if n > 1 else var
    rv = var_unb if running_var is None else mom * np.asarray(running_var, dtype=float).ravel() + (1 - mom) * var_unb
    if rm.size != d or rv.size != d:
        raise ValueError(f"geron_batch_normalization: running statistics must have length {d}")

    return RichResult(
        title="Batch normalization",
        summary_lines=[("Batch size", n), ("Features", d)],
        payload={
            "y": y,
            "x_hat": x_hat,
            "mu": mu,
            "var": var,
            "running_mean": rm,
            "running_var": rv,
            "gamma": np.asarray(g, dtype=float),
            "beta": np.asarray(b, dtype=float),
            "estimate": float(np.mean(y)),
            "n": int(n),
            "method": "Batch normalization (per-feature standardisation then affine rescale)",
        },
    )


def cheatsheet():
    return "hmbntr: Batch normalization: normalize per-batch then affine rescale"
