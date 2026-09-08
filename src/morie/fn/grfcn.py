# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fully convolutional upsampling by transposed convolution."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_fcn_upsample"]

_METHOD = "Transposed convolution (FCN upsampling)"


def geron_fcn_upsample(X, W, stride=2):
    r"""Upsample a feature map with a transposed convolution.

    Every input cell scatters a scaled copy of the kernel into the
    output, overlapping copies adding up:

    .. math::
        Y[i s + p,\; j s + q] \mathrel{+}= X[i, j]\, W[p, q]

    so the output is :math:`(H-1)s + k` tall.  This is the transpose of
    the convolution's gather, which is where the name comes from -- it
    is *not* a deconvolution, and it does not invert anything.

    The overlap is the practical gotcha: when ``stride`` does not divide
    the kernel size, some output cells receive more kernel copies than
    others, which is the origin of the checkerboard artefacts in
    segmentation maps.  ``contribution_counts`` shows exactly which
    cells got how many, so the artefact is visible before it is
    plotted.

    Parameters
    ----------
    X : array-like, shape (H, W_in)
    W : array-like, shape (k_h, k_w)
    stride : int, optional
        Upsampling factor, default 2.

    Returns
    -------
    RichResult
        Payload keys ``output``, ``output_shape``,
        ``contribution_counts``, ``uniform_coverage``,
        ``upsample_factor``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 12, Fully Convolutional Networks section.

    Examples
    --------
    A single cell simply stamps the kernel:

    >>> geron_fcn_upsample([[2.0]], [[1.0, 3.0]], stride=1)["output"]
    [[2.0, 6.0]]

    Stride 2 with a 1x1 kernel spreads the input out, leaving the gaps
    at zero:

    >>> geron_fcn_upsample([[1.0, 2.0]], [[1.0]], stride=2)["output"]
    [[1.0, 0.0, 2.0]]

    A 2x2 kernel at stride 2 tiles perfectly -- every output cell gets
    exactly one contribution, so there is no checkerboard:

    >>> r = geron_fcn_upsample([[1.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]], stride=2)
    >>> r["output"]
    [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]
    >>> r["uniform_coverage"]
    True
    """
    A = np.atleast_2d(np.asarray(X, dtype=float))
    K = np.atleast_2d(np.asarray(W, dtype=float))
    if A.ndim != 2 or K.ndim != 2:
        raise ValueError(f"X and W must both be 2-D, got {A.shape} and {K.shape}.")
    if A.size == 0 or K.size == 0:
        raise ValueError("X and W must be non-empty.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(K)):
        raise ValueError("X and W must be finite.")
    s = int(stride)
    if s < 1:
        raise ValueError(f"stride must be a positive integer, got {s}.")

    H, Wi = A.shape
    kh, kw = K.shape
    oh = (H - 1) * s + kh
    ow = (Wi - 1) * s + kw
    Y = np.zeros((oh, ow))
    counts = np.zeros((oh, ow), dtype=int)
    for i in range(H):
        for j in range(Wi):
            Y[i * s:i * s + kh, j * s:j * s + kw] += A[i, j] * K
            counts[i * s:i * s + kh, j * s:j * s + kw] += 1

    return RichResult(
        title="Transposed convolution (upsample)",
        summary_lines=[("Input", (H, Wi)), ("Output", (oh, ow)), ("Stride", s)],
        payload={
            "output": Y.tolist(),
            "output_shape": (int(oh), int(ow)),
            "contribution_counts": counts.tolist(),
            "uniform_coverage": bool(counts.min() == counts.max()),
            "upsample_factor": float(Y.size) / float(A.size),
            "stride": s,
            "estimate": Y.tolist(),
            "n": int(Y.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grfcn: scatter X[i,j]*W into Y at stride s; uneven overlap = checkerboard artefacts"
