# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Layer normalization."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_layer_normalization"]

_METHOD = "Layer normalization"


def geron_layer_normalization(X, gamma=1.0, beta=0.0, eps=1e-5):
    r"""Normalise across features, one instance at a time.

    .. math::
        \hat x_i = \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}},
        \qquad y = \gamma \hat x + \beta

    The subscript ``i`` is the sample, not the feature: statistics come
    from each row on its own.  That is the whole difference from batch
    norm (:mod:`morie.fn.grbn`), and it is why layer norm works with a
    batch of one and needs no running averages at inference -- which is
    what made it the norm of choice for transformers and RNNs, where the
    batch axis is unreliable.

    ``eps`` sits *inside* the square root, matching the definition; a
    constant row therefore yields zeros rather than a division by zero.

    Parameters
    ----------
    X : array-like, shape (d,) or (m, d)
    gamma : float or array-like, shape (d,), optional
        Learned gain, default 1.
    beta : float or array-like, shape (d,), optional
        Learned shift, default 0.
    eps : float, optional
        Positive stabiliser, default ``1e-5``.

    Returns
    -------
    RichResult
        Payload keys ``output``, ``normalized`` (before gain/shift),
        ``mean``, ``variance``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 11, Layer Normalization section (Ba et al. 2016).

    Examples
    --------
    With ``eps = 0`` the normalised row has mean 0 and variance 1 by
    construction:

    >>> r = geron_layer_normalization([1.0, 3.0], eps=0.0)
    >>> r["normalized"]
    [-1.0, 1.0]
    >>> r["mean"], r["variance"]
    (2.0, 1.0)

    Gain and shift are applied after normalising, so they set the output
    scale directly:

    >>> geron_layer_normalization([1.0, 3.0], gamma=2.0, beta=5.0, eps=0.0)["output"]
    [3.0, 7.0]

    Rows are independent -- doubling one row does not move the other:

    >>> r2 = geron_layer_normalization([[1.0, 3.0], [2.0, 6.0]], eps=0.0)
    >>> r2["normalized"]
    [[-1.0, 1.0], [-1.0, 1.0]]
    """
    A = np.asarray(X, dtype=float)
    single = A.ndim == 1
    if single:
        A = A.reshape(1, -1)
    if A.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got ndim {A.ndim}.")
    if A.shape[1] < 2:
        raise ValueError(
            f"layer norm needs at least 2 features to have a spread, got {A.shape[1]}."
        )
    if not np.all(np.isfinite(A)):
        raise ValueError("X must be finite.")
    eps = float(eps)
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}.")
    d = A.shape[1]
    g = np.asarray(gamma, dtype=float).ravel()
    b = np.asarray(beta, dtype=float).ravel()
    if g.size == 1:
        g = np.full(d, float(g[0]))
    if b.size == 1:
        b = np.full(d, float(b[0]))
    if g.size != d or b.size != d:
        raise ValueError(f"gamma and beta must be scalars or length {d}, got {g.size} and {b.size}.")

    mu = A.mean(axis=1, keepdims=True)
    var = A.var(axis=1, keepdims=True)
    denom = np.sqrt(var + eps)
    if np.any(denom == 0):
        bad = np.flatnonzero(denom.ravel() == 0).tolist()
        raise ValueError(
            f"rows {bad} are constant and eps is 0, so the normaliser divides by "
            f"zero; pass a positive eps."
        )
    Xh = (A - mu) / denom
    Y = g * Xh + b

    return RichResult(
        title="Layer normalization",
        summary_lines=[("Features", int(d)), ("Rows", int(A.shape[0]))],
        payload={
            "output": Y[0].tolist() if single else Y.tolist(),
            "normalized": Xh[0].tolist() if single else Xh.tolist(),
            "mean": float(mu[0, 0]) if single else mu.ravel().tolist(),
            "variance": float(var[0, 0]) if single else var.ravel().tolist(),
            "eps": eps,
            "estimate": Y[0].tolist() if single else Y.tolist(),
            "n": int(A.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grln: per-row (x - mu)/sqrt(var + eps), then gamma/beta -- no batch axis, unlike grbn"
