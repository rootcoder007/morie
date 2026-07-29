# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batch normalization: z-normalize per feature across the mini-batch, then affine scale/shift."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_batch_normalization"]

_METHOD = "Batch normalization (Ioffe & Szegedy 2015)"


def geron_batch_normalization(X, gamma, beta, eps=1e-5, momentum=None,
                              running_mean=None, running_var=None):
    r"""Normalise a mini-batch per feature, then rescale and shift.

    .. math::
        \hat x = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}},
        \qquad y = \gamma \hat x + \beta

    The batch statistics use the **biased** (``ddof=0``) variance, as in
    the paper -- normalising by ``m`` not ``m-1``.  Note what
    :math:`\gamma` and :math:`\beta` buy: without them the layer would be
    forced to hand the next layer a zero-mean unit-variance input, which
    for a sigmoid means being pinned to its linear middle.

    Parameters
    ----------
    X : array-like, shape (m, d)
        Mini-batch, rows are instances.
    gamma, beta : array-like, shape (d,)
        Per-feature scale and shift. Scalars are broadcast.
    eps : float, optional
        Variance floor, default ``1e-5``. Must be positive.
    momentum : float, optional
        If given (in ``[0, 1)``), update the running statistics as
        ``running = momentum*running + (1-momentum)*batch``.
    running_mean, running_var : array-like, optional
        Existing running statistics to update; default zeros and ones.

    Returns
    -------
    RichResult
        Payload keys ``Y``, ``x_hat``, ``batch_mean``, ``batch_var``,
        ``running_mean``, ``running_var``, ``estimate`` (mean of ``Y``),
        ``n``, ``method``.

    References
    ----------
    Géron Ch 11, Batch Normalization section.

    Examples
    --------
    Two instances of one feature, values 0 and 2: mean 1, biased
    variance 1, so the normalised batch is exactly ``[-1, 1]``:

    >>> r = geron_batch_normalization([[0.0], [2.0]], gamma=[1.0], beta=[0.0], eps=0.0)
    >>> r["x_hat"]
    [[-1.0], [1.0]]
    >>> r["batch_mean"], r["batch_var"]
    ([1.0], [1.0])

    gamma and beta act afterwards:

    >>> r2 = geron_batch_normalization([[0.0], [2.0]], gamma=[3.0], beta=[5.0], eps=0.0)
    >>> r2["Y"]
    [[2.0], [8.0]]
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.ndim != 2 or X.size == 0:
        raise ValueError(f"X must be a non-empty 2-D (m, d) array, got shape {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values.")
    m, d = X.shape
    gamma = np.broadcast_to(np.asarray(gamma, dtype=float).ravel(), (d,)).copy() \
        if np.asarray(gamma, dtype=float).size in (1, d) else None
    if gamma is None:
        raise ValueError(f"gamma must have 1 or {d} entries.")
    beta_arr = np.asarray(beta, dtype=float).ravel()
    if beta_arr.size not in (1, d):
        raise ValueError(f"beta must have 1 or {d} entries, got {beta_arr.size}.")
    beta_arr = np.broadcast_to(beta_arr, (d,)).copy()
    eps = float(eps)
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}.")

    mu = X.mean(axis=0)
    var = X.var(axis=0)  # biased, ddof=0 -- as in the paper
    denom = np.sqrt(var + eps)
    if np.any(denom == 0):
        bad = np.flatnonzero(denom == 0).tolist()
        raise ValueError(
            f"features {bad} are constant across the batch and eps=0, so the "
            "normalisation divides by zero; pass eps > 0."
        )
    x_hat = (X - mu) / denom
    Y = gamma * x_hat + beta_arr

    rm = np.zeros(d) if running_mean is None else np.asarray(running_mean, dtype=float).ravel()
    rv = np.ones(d) if running_var is None else np.asarray(running_var, dtype=float).ravel()
    if rm.size != d or rv.size != d:
        raise ValueError(f"running_mean/running_var must have {d} entries.")
    if momentum is not None:
        momentum = float(momentum)
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must lie in [0, 1), got {momentum}.")
        rm = momentum * rm + (1.0 - momentum) * mu
        rv = momentum * rv + (1.0 - momentum) * var

    return RichResult(
        title="Batch normalization",
        summary_lines=[("Batch size", m), ("Features", d)],
        payload={
            "Y": Y.tolist(),
            "x_hat": x_hat.tolist(),
            "batch_mean": mu.tolist(),
            "batch_var": var.tolist(),
            "running_mean": rm.tolist(),
            "running_var": rv.tolist(),
            "eps": eps,
            "estimate": float(Y.mean()),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grbn: batch norm -- x_hat=(x-mu_B)/sqrt(var_B+eps); y=gamma*x_hat+beta"
