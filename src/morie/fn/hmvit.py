# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vision Transformer (ViT): transformer on image patches."""

from . import _array_core as np

from ._richresult import RichResult
from .hmtfm import encoder_params, geron_transformer

__all__ = ["geron_vision_transformer"]


def _lcg(shape, seed, scale=0.1):
    n = int(np.prod(shape))
    s = int(seed) % 2**32
    out = np.empty(n)
    for i in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (2.0 * ((s + 0.5) / 2**32) - 1.0) * scale
    return out.reshape(shape)


def _sinusoidal(T, d):
    pos = np.arange(T).reshape(-1, 1)
    i = np.arange(d).reshape(1, -1)
    angle = pos / np.power(10000.0, (2 * (i // 2)) / float(d))
    pe = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
    return pe


def geron_vision_transformer(image, patch_size, n_layers=2, d_model=8, n_heads=2, n_classes=2, seed=0):
    """
    Vision Transformer (ViT): transformer on image patches.

    Formula: image -> patch embeddings + pos enc -> transformer encoder -> [CLS] classifier

    The pipeline is executed, not described: the image is cut into
    non-overlapping ``patch_size x patch_size`` patches, each patch is
    flattened and linearly embedded, a learned [CLS] token is prepended,
    sinusoidal position encodings are added, the sequence runs through
    :func:`morie.fn.hmtfm.geron_transformer`, and the [CLS] row feeds a
    linear classification head. Parameter counts are resolved exactly,
    reusing :func:`morie.fn.hmtfm.encoder_params` for the encoder part.

    Parameters
    ----------
    image : array-like
        (H, W) or (H, W, C). Both sides must be divisible by `patch_size`.
    patch_size : int
        Patch side length (>= 1).
    n_layers, n_heads : int
        Encoder depth and heads; `n_heads` must divide `d_model`.
    d_model : int, default 8
        Embedding width.
    n_classes : int, default 2
        Output classes (>= 2).
    seed : int, default 0
        LCG seed for the patch embedding, CLS token and head.

    Returns
    -------
    result : RichResult
        Keys: logits, cls, tokens, n_patches, seq_len, total_params,
        estimate, n, method.

    Examples
    --------
    A 4x4 image with 2x2 patches gives 4 patches and a length-5 sequence
    once the [CLS] token is prepended:

    >>> img = [[float(i * 4 + j) for j in range(4)] for i in range(4)]
    >>> r = geron_vision_transformer(img, patch_size=2, n_layers=1, d_model=4, n_heads=2, n_classes=3)
    >>> int(r["n_patches"]), int(r["seq_len"])
    (4, 5)
    >>> r["logits"].shape
    (3,)
    >>> int(r["patch_dim"])
    4
    >>> int(r["total_params"]) == 4 * 4 + 4 + 4 + 5 * 4 + int(r["encoder_params"]) + 4 * 3 + 3
    True

    References
    ----------
    Géron Ch 16
    """
    img = np.asarray(image, dtype=float)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.ndim != 3 or img.size == 0:
        raise ValueError("geron_vision_transformer: image must be (H, W) or (H, W, C) and non-empty")
    if not np.all(np.isfinite(img)):
        raise ValueError("geron_vision_transformer: image contains non-finite values")
    H, W, C = img.shape
    p = int(patch_size)
    if p < 1:
        raise ValueError(f"geron_vision_transformer: patch_size must be >= 1, got {p}")
    if H % p or W % p:
        raise ValueError(
            f"geron_vision_transformer: patch_size {p} does not tile a {H}x{W} image; "
            "ViT requires H and W divisible by the patch size"
        )
    d = int(d_model)
    h = int(n_heads)
    if d < 1 or h < 1 or d % h:
        raise ValueError(f"geron_vision_transformer: need d_model % n_heads == 0, got d_model={d}, n_heads={h}")
    K = int(n_classes)
    if K < 2:
        raise ValueError(f"geron_vision_transformer: n_classes must be >= 2, got {K}")
    L = int(n_layers)
    if L < 1:
        raise ValueError(f"geron_vision_transformer: n_layers must be >= 1, got {L}")

    nh, nw = H // p, W // p
    n_patches = nh * nw
    patch_dim = p * p * C
    patches = np.empty((n_patches, patch_dim))
    idx = 0
    for i in range(nh):
        for j in range(nw):
            patches[idx] = img[i * p : (i + 1) * p, j * p : (j + 1) * p, :].reshape(-1)
            idx += 1

    E = _lcg((patch_dim, d), int(seed) + 1)
    b_e = _lcg((d,), int(seed) + 2)
    cls = _lcg((1, d), int(seed) + 3)
    tokens = np.vstack([cls, patches @ E + b_e])
    seq_len = tokens.shape[0]
    tokens = tokens + _sinusoidal(seq_len, d)

    enc = geron_transformer(tokens, n_heads=h, n_layers=L, seed=int(seed) + 10)
    Y = np.asarray(enc["Y"], dtype=float)
    Wh = _lcg((d, K), int(seed) + 4)
    b_h = _lcg((K,), int(seed) + 5)
    logits = Y[0] @ Wh + b_h

    enc_p = encoder_params(d, 4 * d, L)
    total = patch_dim * d + d + d + seq_len * d + enc_p + d * K + K

    return RichResult(
        title="Vision Transformer",
        summary_lines=[
            ("Patches", n_patches),
            ("Sequence length", seq_len),
            ("Patch dim", patch_dim),
            ("d_model", d),
            ("Parameters", int(total)),
        ],
        interpretation=(
            "ViT has no convolutional inductive bias: spatial structure enters only through the patch "
            "grid and the position encodings, which is why it needs more data than a CNN of similar size."
        ),
        payload={
            "logits": logits,
            "cls": Y[0],
            "tokens": Y,
            "patches": patches,
            "n_patches": int(n_patches),
            "seq_len": int(seq_len),
            "patch_dim": int(patch_dim),
            "grid": (int(nh), int(nw)),
            "encoder_params": int(enc_p),
            "total_params": int(total),
            "predicted": int(np.argmax(logits)),
            "estimate": float(total),
            "n": int(n_patches),
            "method": "ViT: patch embedding + [CLS] + sinusoidal positions + transformer encoder (hmtfm)",
        },
    )


def cheatsheet():
    return "hmvit: Vision Transformer (ViT): transformer on image patches"
