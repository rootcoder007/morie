# morie.fn -- function file (rootcoder007/morie)
"""Stochastic-gradient update on a mini-batch."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["sgd_update"]


def sgd_update(beta, batch_grads, eta=0.01):
    r"""One SGD step using the mean gradient over a mini-batch.

    Given per-example gradients :math:`g_i` for the :math:`B` examples in the
    batch,

    .. math::
        \beta_{t+1} = \beta_t - \eta\, \frac{1}{B}\sum_{i=1}^{B} g_i.

    Averaging rather than summing is what keeps the step size independent of
    the batch size. The returned ``grad_se`` is the standard error of that
    mean across the batch -- a cheap read on how noisy the step was.

    Parameters
    ----------
    beta : array-like
        Current parameters, shape ``(p,)``.
    batch_grads : array-like
        Per-example gradients, shape ``(B, p)``. A 1-D input is treated as a
        single example.
    eta : float
        Learning rate. Must be positive.

    Returns
    -------
    RichResult
        ``beta`` (updated), ``grad_mean``, ``grad_se``, ``batch_size``,
        ``update``.

    References
    ----------
    Robbins, H., & Monro, S. (1951). A stochastic approximation method.
        *Annals of Mathematical Statistics*, 22(3), 400-407.

    Examples
    --------
    >>> import numpy as np
    >>> r = sgd_update([0.0, 0.0], [[1.0, 2.0], [3.0, 4.0]], eta=0.1)
    >>> [float(v) for v in r["grad_mean"]]
    [2.0, 3.0]
    >>> [float(round(v, 6)) for v in r["beta"]]
    [-0.2, -0.3]

    Averaging keeps the step independent of batch size -- duplicating the
    batch does not change the update.

    >>> r2 = sgd_update([0.0, 0.0], [[1.0, 2.0], [3.0, 4.0]] * 4, eta=0.1)
    >>> bool(np.allclose(r["beta"], r2["beta"]))
    True
    """
    if eta <= 0:
        raise ValueError("eta must be positive")
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    G = np.atleast_2d(np.asarray(batch_grads, dtype=float))
    if G.shape[1] != beta.size:
        raise ValueError(f"batch_grads has {G.shape[1]} columns but beta has {beta.size} entries")
    B = G.shape[0]
    gbar = G.mean(axis=0)
    se = G.std(axis=0, ddof=1) / np.sqrt(B) if B > 1 else np.full(beta.size, np.nan)
    update = -eta * gbar
    return RichResult(
        title="SGD mini-batch update",
        summary_lines=[("batch size", B), ("eta", float(eta))],
        payload={
            "beta": beta + update,
            "update": update,
            "grad_mean": gbar,
            "grad_se": se,
            "batch_size": int(B),
            "eta": float(eta),
            "method": "sgd_update",
        },
    )


def cheatsheet():
    return "sgdup: averages per-example gradients so the step is batch-size invariant; grad_se reports the noise"
