# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Layer normalization: normalize across features within a single sample."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_layer_normalization"]

_METHOD = "Layer normalization (per-sample, across features)"


def geron_layer_normalization(x, gamma=1.0, beta=0.0, eps=1e-5):
    """
    Layer normalization: normalize across features within a single sample.

    Formula: x_hat = (x - mu) / sqrt(var + eps); y = gamma * x_hat + beta

    The statistics are taken **along the feature axis of each row**, not
    down the batch as in batch normalization
    (:func:`morie.fn.hmbntr.geron_batch_normalization`).  That single
    difference is why layer norm works with a batch of one and needs no
    separate inference-time running statistics: every sample is
    self-contained.  ``gamma`` and ``beta`` are still per-feature, so
    the affine step can undo the normalization if the network wants it.

    ``var`` uses the biased (population) denominator ``1/d``, which is
    the convention the formula's ``sqrt(var + eps)`` assumes.

    :func:`morie.fn.km018.kamath_ch2_layer_norm` covers the same
    operation in Kamath's single-vector notation, with a scalar gain and
    pinnable mu/sigma; this entry is the batched Géron form with beta
    and the eps floor inside the square root.

    Parameters
    ----------
    x : array-like, shape (m, d) or (d,)
        Rows are samples, columns features.
    gamma, beta : float or array-like of length d
        Per-feature scale and shift.
    eps : float
        Variance floor, must be positive -- it is what keeps a constant
        row from dividing by zero.

    Returns
    -------
    result : RichResult
        Keys: y, x_hat, mu, var, estimate, n, method.

    Examples
    --------
    Row ``[1, 3]`` has mean 2 and population variance 1, so with eps=0
    it normalizes to ``[-1, 1]``:

    >>> r = geron_layer_normalization([[1.0, 3.0]], eps=0.0)
    >>> [float(v) for v in r["x_hat"].ravel()]
    [-1.0, 1.0]
    >>> float(r["mu"][0]), float(r["var"][0])
    (2.0, 1.0)

    Each row is normalized independently -- the second row has a
    completely different scale and still comes out as ``[-1, 1]``:

    >>> r2 = geron_layer_normalization([[1.0, 3.0], [100.0, 300.0]], eps=0.0)
    >>> [[round(float(v), 9) for v in row] for row in r2["x_hat"]]
    [[-1.0, 1.0], [-1.0, 1.0]]

    The affine step is applied per feature:

    >>> a = geron_layer_normalization([[1.0, 3.0]], gamma=[2.0, 3.0], beta=[5.0, -5.0], eps=0.0)
    >>> [float(v) for v in a["y"].ravel()]
    [3.0, -2.0]

    A constant row has zero variance; eps keeps it finite rather than
    producing NaN:

    >>> c = geron_layer_normalization([[7.0, 7.0]], eps=1e-5)
    >>> [float(v) for v in c["x_hat"].ravel()]
    [0.0, 0.0]

    References
    ----------
    Géron Ch 11
    """
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"geron_layer_normalization: x must be 1-D or 2-D, got ndim={X.ndim}")
    if X.size == 0:
        raise ValueError("geron_layer_normalization: x is empty")
    if not np.all(np.isfinite(X)):
        raise ValueError("geron_layer_normalization: x contains non-finite values")
    m, d = X.shape
    if d < 2:
        raise ValueError(
            f"geron_layer_normalization: layer norm averages across features, but x has only {d} feature; "
            f"with one feature every row normalizes to 0"
        )
    e = float(eps)
    if e < 0:
        raise ValueError(f"geron_layer_normalization: eps must be non-negative, got {eps!r}")

    g = np.broadcast_to(np.atleast_1d(np.asarray(gamma, dtype=float)), (d,)).astype(float)
    b = np.broadcast_to(np.atleast_1d(np.asarray(beta, dtype=float)), (d,)).astype(float)

    mu = X.mean(axis=1)
    var = X.var(axis=1)
    if e == 0 and np.any(var == 0):
        j = int(np.flatnonzero(var == 0)[0])
        raise ValueError(
            f"geron_layer_normalization: row {j} is constant (variance 0) and eps is 0, "
            f"so the normalization divides by zero; pass a positive eps"
        )
    x_hat = (X - mu[:, None]) / np.sqrt(var[:, None] + e)
    y = g * x_hat + b

    return RichResult(
        title="Layer normalization",
        summary_lines=[("Samples", int(m)), ("Features", int(d)), ("eps", e)],
        interpretation=(
            "Statistics come from the sample itself, so batch size is irrelevant and no running "
            "statistics are needed at inference -- the reason RNNs and transformers use it."
        ),
        payload={
            "y": y,
            "x_hat": x_hat,
            "mu": mu,
            "var": var,
            "gamma": g,
            "beta": b,
            "estimate": float(np.mean(y)),
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlntr: layer norm across features per sample; y = gamma*(x-mu)/sqrt(var+eps) + beta"
