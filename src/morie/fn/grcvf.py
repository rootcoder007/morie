# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""2D convolution forward pass with a single filter, stride s and padding p."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_conv2d_forward"]

_METHOD = "2-D convolution forward pass"


def _pair(v, name):
    a = np.atleast_1d(np.asarray(v))
    try:
        ai = a.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer or pair of integers.") from exc
    if not np.array_equal(ai, a):
        raise ValueError(f"{name} must be whole numbers, got {v!r}.")
    if ai.size == 1:
        ai = np.repeat(ai, 2)
    if ai.size != 2:
        raise ValueError(f"{name} must have 1 or 2 entries, got {ai.size}.")
    return int(ai[0]), int(ai[1])


def geron_conv2d_forward(X, W, b=0.0, stride=1, padding=0):
    r"""Cross-correlate a single filter over a (possibly multi-channel) map.

    .. math::
        Y[i,j] = \sum_{c,u,v} W[c,u,v]\, X[c,\,is+u-p,\,js+v-p] + b

    As in every deep-learning library this is cross-correlation, not the
    flipped convolution of signal processing -- the distinction does not
    matter when the kernel is learned, but it does when you hand-build
    one, as the doctest below does.

    Parameters
    ----------
    X : array-like, shape (H, W) or (C, H, W)
        Input map.
    W : array-like, shape (kh, kw) or (C, kh, kw)
        Filter; the channel count must match ``X``.
    b : float, optional
        Bias added to every output cell.
    stride : int or (sh, sw), optional
    padding : int or (ph, pw), optional
        Zero padding on each side.

    Returns
    -------
    RichResult
        Payload keys ``Y``, ``out_shape``, ``n_multiply_adds``,
        ``padded_shape``, ``estimate`` (mean of ``Y``), ``n``,
        ``method``.

    References
    ----------
    Géron Ch 12, Convolutional Layers section.

    Examples
    --------
    A diagonal ``2x2`` filter on a ``3x3`` ramp sums the two diagonal
    neighbours of each window:

    >>> X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    >>> r = geron_conv2d_forward(X, [[1.0, 0.0], [0.0, 1.0]])
    >>> r["Y"]
    [[6.0, 8.0], [12.0, 14.0]]
    >>> r["out_shape"]
    (2, 2)

    Stride 2 keeps only the top-left window; padding 1 restores 3x3:

    >>> geron_conv2d_forward(X, [[1.0, 0.0], [0.0, 1.0]], stride=2)["Y"]
    [[6.0]]
    >>> geron_conv2d_forward(X, [[1.0]], padding=1)["out_shape"]
    (5, 5)
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    if X.ndim == 2:
        X = X[None, :, :]
    if W.ndim == 2:
        W = W[None, :, :]
    if X.ndim != 3 or W.ndim != 3:
        raise ValueError(
            f"X and W must be 2-D or 3-D, got ndim {X.ndim} and {W.ndim}."
        )
    if X.shape[0] != W.shape[0]:
        raise ValueError(
            f"channel mismatch: X has {X.shape[0]} channels, W has {W.shape[0]}."
        )
    if X.size == 0 or W.size == 0:
        raise ValueError("X and W must be non-empty.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(W)):
        raise ValueError("X and W must be finite.")
    b = float(b)
    if not np.isfinite(b):
        raise ValueError(f"b must be finite, got {b}.")
    sh, sw = _pair(stride, "stride")
    ph, pw = _pair(padding, "padding")
    if sh < 1 or sw < 1:
        raise ValueError(f"stride must be positive, got {(sh, sw)}.")
    if ph < 0 or pw < 0:
        raise ValueError(f"padding must be non-negative, got {(ph, pw)}.")

    C, H, Wd = X.shape
    kh, kw = W.shape[1], W.shape[2]
    Xp = np.pad(X, ((0, 0), (ph, ph), (pw, pw)), mode="constant")
    Hp, Wp = Xp.shape[1], Xp.shape[2]
    oh = (Hp - kh) // sh + 1
    ow = (Wp - kw) // sw + 1
    if Hp < kh or Wp < kw:
        raise ValueError(
            f"filter {kh}x{kw} does not fit the padded input {Hp}x{Wp}."
        )

    Y = np.empty((oh, ow), dtype=float)
    for i in range(oh):
        r0 = i * sh
        for j in range(ow):
            c0 = j * sw
            Y[i, j] = float(np.sum(Xp[:, r0:r0 + kh, c0:c0 + kw] * W)) + b

    return RichResult(
        title="Conv2D forward",
        summary_lines=[("Output shape", (oh, ow)), ("Channels", C)],
        payload={
            "Y": Y.tolist(),
            "out_shape": (int(oh), int(ow)),
            "padded_shape": (int(Hp), int(Wp)),
            "n_multiply_adds": int(oh * ow * C * kh * kw),
            "stride": (sh, sw),
            "padding": (ph, pw),
            "estimate": float(Y.mean()),
            "n": int(Y.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grcvf: conv2d forward Y[i,j] = sum_{c,u,v} W*X[c, i*s+u-p, j*s+v-p] + b"
