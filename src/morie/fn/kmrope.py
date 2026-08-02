# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rotary positional embedding (RoPE) at arbitrary positions."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_rotary_positional_embedding"]


def kamath_rotary_positional_embedding(q, positions=None, base=10000.0):
    """Rotate each feature pair by m * theta_i:

    [q_2i, q_2i+1] -> [cos(m t_i) q_2i - sin(m t_i) q_2i+1,
                       sin(m t_i) q_2i + cos(m t_i) q_2i+1],
    with t_i = base^(-2i/d).

    ``positions`` is explicit, which is the point of this variant --
    KV-cache decoding, packed sequences and context-extension
    experiments all rotate rows by positions that are not
    ``0, 1, 2, ...``. ``morie.fn.rotrp`` covers the contiguous case;
    with ``positions=None`` this reduces to it, and the test file
    asserts they agree rather than trusting that they do.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, RoPE (Su et al.
    2021).

    Examples
    --------
    >>> out = kamath_rotary_positional_embedding([[1.0, 0.0]], [0])
    >>> out["y"]
    [[1.0, 0.0]]
    >>> import math
    >>> one = kamath_rotary_positional_embedding([[1.0, 0.0]], [1])
    >>> abs(one["y"][0][0] - math.cos(1.0)) < 1e-12
    True
    >>> abs(one["y"][0][1] - math.sin(1.0)) < 1e-12
    True
    >>> big = kamath_rotary_positional_embedding([[1.0, 0.0, 1.0, 0.0]], [1])
    >>> abs(big["angles"][0][1] - 1 / 100.0) < 1e-12
    True
    """
    x = np.asarray(q, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("q must be (d,) or (seq_len, d).")
    T, d = x.shape
    if T == 0 or d == 0:
        raise ValueError("q is empty.")
    if d % 2 != 0:
        raise ValueError(
            f"RoPE rotates feature PAIRS, so d must be even; got {d}.")
    base = float(base)
    if base <= 1.0:
        raise ValueError(
            f"base must exceed 1; got {base}. At base 1 every "
            "frequency is identical and RoPE stops encoding position.")
    if positions is None:
        m = np.arange(T, dtype=float)
    else:
        m = np.atleast_1d(np.asarray(positions, dtype=float)).ravel()
        if m.size != T:
            raise ValueError(
                f"got {m.size} positions for {T} rows.")
        if np.any(m < 0):
            raise ValueError("positions must be non-negative.")
    half = d // 2
    i = np.arange(half, dtype=float)
    theta = base ** (-2.0 * i / d)
    angles = m[:, None] * theta[None, :]
    cos, sin = np.cos(angles), np.sin(angles)
    even, odd = x[:, 0::2], x[:, 1::2]
    y = np.empty_like(x)
    y[:, 0::2] = even * cos - odd * sin
    y[:, 1::2] = even * sin + odd * cos
    return RichResult(payload={
        "y": [[float(v) for v in row] for row in y],
        "angles": [[float(v) for v in row] for row in angles],
        "theta": [float(v) for v in theta],
        "positions": [float(v) for v in m],
        "estimate": float(y[0, 0]),
        "base": base, "d": d, "n": T,
        "method": "Rotary positional embedding at explicit positions"})


def cheatsheet():
    return "kmrope: rotate pairs by m*base^(-2i/d); positions explicit"
