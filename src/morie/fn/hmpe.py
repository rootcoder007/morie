# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sinusoidal positional encoding."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_positional_encoding"]


def geron_positional_encoding(pos, d_model, base=10000.0):
    """
    Sinusoidal positional encoding.

    Formula: PE_{pos,2i} = sin(pos/10000^{2i/d});
    PE_{pos,2i+1} = cos(pos/10000^{2i/d})

    Attention is permutation-equivariant, so order has to be injected
    into the token vectors themselves. The sinusoids do it with no
    parameters and with a property learned embeddings do not have: for a
    fixed offset k, PE(pos+k) is a FIXED linear map of PE(pos) (a
    rotation of each frequency pair), so relative position is linearly
    recoverable at any distance, including lengths never seen in
    training. ``rotation_check`` is the largest deviation from that
    identity over the returned positions.

    Parameters
    ----------
    pos : int or array-like
        Position index, or an array of them.
    d_model : int
        Embedding width; must be even so the sin/cos pairs line up.
    base : float, default 10000.0
        Wavelength base.

    Returns
    -------
    result : RichResult
        Keys: pe, wavelengths, rotation_check, estimate, n, method.

    Examples
    --------
    At position 0 every sine is 0 and every cosine is 1:

    >>> [float(v) for v in geron_positional_encoding(0, 4)["pe"]]
    [0.0, 1.0, 0.0, 1.0]

    With d_model = 2 the only frequency is 1, so position 1 gives
    (sin 1, cos 1):

    >>> [round(float(v), 6) for v in geron_positional_encoding(1, 2)["pe"]]
    [0.841471, 0.540302]

    Rows are unit-norm in each sin/cos pair, so ||PE|| = sqrt(d/2):

    >>> round(float(np.linalg.norm(geron_positional_encoding(7, 8)["pe"])), 6)
    2.0

    References
    ----------
    Geron Ch 15
    """
    d = int(d_model)
    if d <= 0 or d % 2 != 0:
        raise ValueError(f"geron_positional_encoding: d_model must be a positive even integer, got {d_model!r}")
    b = float(base)
    if not np.isfinite(b) or b <= 1.0:
        raise ValueError(f"geron_positional_encoding: base must be > 1, got {base!r}")
    scalar = np.ndim(pos) == 0
    p = np.atleast_1d(np.asarray(pos, dtype=float)).ravel()
    if p.size == 0:
        raise ValueError("geron_positional_encoding: pos is empty")
    if not np.all(np.isfinite(p)):
        raise ValueError("geron_positional_encoding: pos contains non-finite values")

    i = np.arange(d // 2, dtype=float)
    inv = b ** (-2.0 * i / d)
    ang = p[:, None] * inv[None, :]
    pe = np.empty((p.size, d), dtype=float)
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)

    # PE(pos+1) = R PE(pos) for a fixed block-rotation R: check it.
    c, s = np.cos(inv), np.sin(inv)
    shifted = np.empty_like(pe)
    shifted[:, 0::2] = pe[:, 0::2] * c + pe[:, 1::2] * s
    shifted[:, 1::2] = pe[:, 1::2] * c - pe[:, 0::2] * s
    direct = np.empty_like(pe)
    ang1 = (p[:, None] + 1.0) * inv[None, :]
    direct[:, 0::2] = np.sin(ang1)
    direct[:, 1::2] = np.cos(ang1)
    rot = float(np.max(np.abs(shifted - direct)))

    out = pe[0] if scalar else pe
    return RichResult(
        title="Sinusoidal positional encoding",
        summary_lines=[("d_model", d), ("Positions", int(p.size)), ("Rotation identity error", rot)],
        interpretation="Relative offsets are a fixed linear map of the encoding, so it extrapolates past training length.",
        payload={
            "pe": out,
            "wavelengths": 2.0 * np.pi / inv,
            "rotation_check": rot,
            "d_model": d,
            "estimate": out,
            "n": int(p.size),
            "method": "Sinusoidal positional encoding (Vaswani et al. form)",
        },
    )


def cheatsheet():
    return "hmpe: Sinusoidal positional encoding"
