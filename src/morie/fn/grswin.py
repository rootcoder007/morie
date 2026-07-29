# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Swin Transformer: self-attention restricted to non-overlapping local windows."""

import numpy as np

from ._richresult import RichResult
from .grsdpa import attend

__all__ = ["geron_swin_window_attention"]

_METHOD = "Swin windowed self-attention"


def geron_swin_window_attention(X, window_size, WQ, WK, WV, shift=0):
    r"""Attend within ``M x M`` windows instead of globally.

    Partition the ``H x W`` feature map into non-overlapping windows and
    run ordinary self-attention inside each.  Global attention costs
    :math:`O((HW)^2)`; this costs :math:`O(HW M^2)` -- linear in the
    number of tokens for fixed ``M``, which is what makes Swin usable at
    image resolutions.

    The price is that windows never talk to each other.  Swin's answer is
    to *shift* the partition by ``M/2`` on alternating blocks, so the
    boundaries move and information crosses; ``shift`` implements that
    cyclic roll.  Attention itself is delegated to
    :func:`morie.fn.grsdpa.attend`.

    Parameters
    ----------
    X : array-like, shape (H, W, d_model)
    window_size : int
        Must divide both H and W.
    WQ, WK : array-like, shape (d_model, d_k)
    WV : array-like, shape (d_model, d_v)
    shift : int, optional
        Cyclic shift applied before partitioning (Swin uses
        ``window_size // 2`` on every second block).

    Returns
    -------
    RichResult
        Payload keys ``output`` (H x W x d_v), ``window_weights``,
        ``n_windows``, ``tokens_per_window``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 16, Swin Transformer section.

    Examples
    --------
    A 2x2 map with window size 1: each token can only attend to itself,
    so the output is exactly ``X @ WV``.

    >>> X = [[[1.0], [2.0]], [[3.0], [4.0]]]
    >>> I = [[1.0]]
    >>> r = geron_swin_window_attention(X, 1, I, I, I)
    >>> r["n_windows"], r["tokens_per_window"]
    (4, 1)
    >>> r["output"]
    [[[1.0], [2.0]], [[3.0], [4.0]]]

    One window covering everything is plain global self-attention, and
    the four values get mixed:

    >>> r2 = geron_swin_window_attention(X, 2, I, I, I)
    >>> r2["n_windows"]
    1
    >>> round(r2["output"][0][0][0], 6)
    3.492653
    """
    A = np.asarray(X, dtype=float)
    if A.ndim != 3 or A.size == 0:
        raise ValueError(f"X must be a non-empty (H, W, d_model) array, got shape {A.shape}.")
    if not np.all(np.isfinite(A)):
        raise ValueError("X contains non-finite values.")
    H, W, d = A.shape
    M = int(window_size)
    if M < 1:
        raise ValueError(f"window_size must be at least 1, got {M}.")
    if H % M or W % M:
        raise ValueError(f"window_size {M} does not divide the {H}x{W} feature map.")
    mats = []
    for name, Wm in (("WQ", WQ), ("WK", WK), ("WV", WV)):
        Mm = np.atleast_2d(np.asarray(Wm, dtype=float))
        if Mm.shape[0] != d:
            raise ValueError(f"{name} must have {d} rows to match d_model, got {Mm.shape[0]}.")
        mats.append(Mm)
    Wq, Wk, Wv = mats
    if Wq.shape[1] != Wk.shape[1]:
        raise ValueError(f"WQ maps to d_k={Wq.shape[1]} but WK maps to {Wk.shape[1]}.")
    shift = int(shift)
    if not (0 <= shift < M) and M > 1:
        raise ValueError(f"shift must lie in [0, {M - 1}], got {shift}.")

    B = np.roll(A, shift=(-shift, -shift), axis=(0, 1)) if shift else A
    out = np.zeros((H, W, Wv.shape[1]))
    weights = []
    for i in range(0, H, M):
        for j in range(0, W, M):
            win = B[i : i + M, j : j + M, :].reshape(M * M, d)
            o, w = attend(win @ Wq, win @ Wk, win @ Wv)
            out[i : i + M, j : j + M, :] = o.reshape(M, M, Wv.shape[1])
            weights.append(w.tolist())
    if shift:
        out = np.roll(out, shift=(shift, shift), axis=(0, 1))

    return RichResult(
        title="Swin window attention",
        summary_lines=[("Windows", len(weights)), ("Window size", M), ("Shift", shift)],
        payload={
            "output": out.tolist(),
            "window_weights": weights,
            "n_windows": len(weights),
            "tokens_per_window": int(M * M),
            "shift": shift,
            "estimate": out.tolist(),
            "n": int(H * W),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grswin: attention inside MxM windows, O(HW M^2) not O((HW)^2); shift= rolls the partition"
