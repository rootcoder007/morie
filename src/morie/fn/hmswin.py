# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Swin Transformer: shifted-window attention."""

from . import _array_core as np

from ._richresult import RichResult
from .hmsdp import geron_scaled_dot_product

__all__ = ["geron_swin"]


def _lcg(shape, seed, scale=0.1):
    n = int(np.prod(shape))
    s = int(seed) % 2**32
    out = np.empty(n)
    for i in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (2.0 * ((s + 0.5) / 2**32) - 1.0) * scale
    return out.reshape(shape)


def geron_swin(image, window_size, n_layers=2, d_model=4, seed=0):
    """
    Swin Transformer: shifted-window attention.

    Formula: local attention within windows; shift windows between blocks

    Real windowed attention. Each pixel becomes a token (linearly embedded
    to `d_model`), the token grid is partitioned into non-overlapping
    ``window_size x window_size`` windows, and attention runs *inside*
    each window via :func:`morie.fn.hmsdp.geron_scaled_dot_product`. Every
    odd-numbered layer cyclically shifts the grid by ``window_size // 2``
    before partitioning, which is exactly what lets information cross
    window boundaries; the shift is undone afterwards so the output stays
    registered with the input. Cost is linear in the number of tokens
    (``n_windows * window_size^4``) instead of quadratic.

    Parameters
    ----------
    image : array-like
        (H, W) or (H, W, C). Both sides must be divisible by `window_size`.
    window_size : int
        Window side length (>= 1).
    n_layers : int, default 2
        Blocks; even indices use regular windows, odd indices shifted ones.
    d_model : int, default 4
        Token embedding width.
    seed : int, default 0
        LCG seed for the embedding and per-layer projections.

    Returns
    -------
    result : RichResult
        Keys: Y, pooled, n_windows, shifted_layers, attention_pairs,
        estimate, n, method.

    Examples
    --------
    A 4x4 image with 2x2 windows has 4 windows of 4 tokens each:

    >>> img = [[float(i * 4 + j) for j in range(4)] for i in range(4)]
    >>> r = geron_swin(img, window_size=2, n_layers=1, d_model=4)
    >>> int(r["n_windows"]), int(r["window_tokens"])
    (4, 4)
    >>> r["Y"].shape
    (4, 4, 4)
    >>> int(r["shifted_layers"])
    0

    Two layers introduce one shifted block, which is what connects tokens
    that a single non-shifted layer keeps apart:

    >>> int(geron_swin(img, window_size=2, n_layers=2, d_model=4)["shifted_layers"])
    1

    References
    ----------
    Géron Ch 16
    """
    img = np.asarray(image, dtype=float)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.ndim != 3 or img.size == 0:
        raise ValueError("geron_swin: image must be (H, W) or (H, W, C) and non-empty")
    if not np.all(np.isfinite(img)):
        raise ValueError("geron_swin: image contains non-finite values")
    H, W, C = img.shape
    w = int(window_size)
    if w < 1:
        raise ValueError(f"geron_swin: window_size must be >= 1, got {w}")
    if H % w or W % w:
        raise ValueError(
            f"geron_swin: window_size {w} does not tile a {H}x{W} grid; Swin needs H and W divisible by it"
        )
    d = int(d_model)
    if d < 1:
        raise ValueError(f"geron_swin: d_model must be >= 1, got {d}")
    L = int(n_layers)
    if L < 1:
        raise ValueError(f"geron_swin: n_layers must be >= 1, got {L}")

    E = _lcg((C, d), int(seed) + 1)
    X = img.reshape(-1, C) @ E
    X = X.reshape(H, W, d)

    shift = w // 2
    shifted_layers = 0
    for layer in range(L):
        do_shift = (layer % 2 == 1) and shift > 0
        shifted_layers += int(do_shift)
        Z = np.roll(X, shift=(-shift, -shift), axis=(0, 1)) if do_shift else X
        base = int(seed) + 100 * (layer + 1)
        Wq, Wk, Wv = (_lcg((d, d), base + i) for i in range(1, 4))
        out = np.empty_like(Z)
        for i0 in range(0, H, w):
            for j0 in range(0, W, w):
                blk = Z[i0 : i0 + w, j0 : j0 + w, :].reshape(-1, d)
                a = geron_scaled_dot_product(blk @ Wq, blk @ Wk, blk @ Wv, d_k=d)
                out[i0 : i0 + w, j0 : j0 + w, :] = np.asarray(a["Y"], dtype=float).reshape(w, w, d)
        Z = Z + out
        X = np.roll(Z, shift=(shift, shift), axis=(0, 1)) if do_shift else Z

    n_windows = (H // w) * (W // w)
    tokens_per_window = w * w
    pairs = n_windows * tokens_per_window * tokens_per_window

    return RichResult(
        title="Swin Transformer (shifted windows)",
        summary_lines=[
            ("Grid", f"{H}x{W}"),
            ("Windows", n_windows),
            ("Tokens per window", tokens_per_window),
            ("Shifted blocks", shifted_layers),
            ("Attention pairs per layer", pairs),
        ],
        interpretation=(
            "Windowed attention costs n_windows * window^4 pairs per layer instead of (H*W)^2; "
            "shifting every other block is what stops the windows from becoming isolated."
        ),
        payload={
            "Y": X,
            "pooled": X.reshape(-1, d).mean(axis=0),
            "n_windows": int(n_windows),
            "window_tokens": int(tokens_per_window),
            "shifted_layers": int(shifted_layers),
            "attention_pairs": int(pairs),
            "full_attention_pairs": int((H * W) ** 2),
            "estimate": float(pairs),
            "n": int(H * W),
            "method": "Swin: window-local scaled dot-product attention with alternating cyclic shift",
        },
    )


def cheatsheet():
    return "hmswin: Swin Transformer: shifted-window attention"


# compact alias per ledger/NAMING.md
geronswin = geron_swin
