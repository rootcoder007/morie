# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""YaRN: NTK-aware RoPE frequency rescaling for context
extrapolation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_yarn_context_extrapolation"]


def kamath_yarn_context_extrapolation(theta, scale, d, ramp=None):
    """theta_i_new = theta_i * (1 / s^(2i/d)).

    NTK-aware means the rescaling is NOT uniform: the highest
    frequency (i = 0) is left untouched and the lowest is divided by
    the full factor s. Stretching every frequency equally is position
    interpolation, and it destroys the high-frequency detail the model
    uses for local order -- the exponent 2i/d is what avoids that.

    ``theta`` is either the RoPE base (a scalar, from which
    theta_i = base^(-2i/d) is built) or the d/2 frequencies
    themselves. ``ramp`` optionally blends scaled and unscaled
    frequencies over an index band ``(lo, hi)``: below lo the original
    frequency is kept, above hi the scaled one, linearly in between.
    The spec line gives no numeric band, so there is no default and
    nothing is blended unless the caller asks.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 10,
    YaRN; that section is not in the 2024 PDF, so the rescaling is
    implemented exactly as the spec line states (Peng et al. 2023).

    Examples
    --------
    >>> out = kamath_yarn_context_extrapolation(10000.0, 2.0, 4)
    >>> out["theta"]
    [1.0, 0.01]
    >>> abs(out["theta_new"][0] - 1.0) < 1e-12
    True
    >>> abs(out["theta_new"][1] - 0.01 / 2 ** 0.5) < 1e-12
    True
    >>> out["effective_context_multiplier"]
    2.0
    """
    d = int(d)
    s = float(scale)
    if d < 2 or d % 2 != 0:
        raise ValueError(
            f"d must be a positive even embedding width; got {d}.")
    if s <= 0:
        raise ValueError(f"the scale factor must be positive; got {s}.")
    half = d // 2
    t = np.asarray(theta, dtype=float)
    if t.ndim == 0:
        base = float(t)
        if base <= 1.0:
            raise ValueError(
                f"a scalar theta is the RoPE base and must exceed 1; "
                f"got {base}.")
        i = np.arange(half, dtype=float)
        freqs = base ** (-2.0 * i / d)
    else:
        freqs = t.ravel()
        if freqs.size != half:
            raise ValueError(
                f"an array theta must hold the d/2 = {half} "
                f"frequencies; got {freqs.size}.")
        if np.any(freqs <= 0):
            raise ValueError("RoPE frequencies must be positive.")
    i = np.arange(half, dtype=float)
    factor = s ** (-2.0 * i / d)
    new = freqs * factor
    if ramp is not None:
        lo, hi = (float(v) for v in ramp)
        if not 0 <= lo < hi <= half:
            raise ValueError(
                f"ramp must be (lo, hi) with 0 <= lo < hi <= {half}.")
        w = np.clip((i - lo) / (hi - lo), 0.0, 1.0)
        new = (1.0 - w) * freqs + w * new
    return RichResult(payload={
        "theta": [float(v) for v in freqs],
        "theta_new": [float(v) for v in new],
        "scale_factors": [float(v) for v in factor],
        "effective_context_multiplier": s,
        "ramp": None if ramp is None else (float(ramp[0]), float(ramp[1])),
        "estimate": float(new[-1]),
        "d": d, "n": half,
        "method": "YaRN NTK-aware RoPE frequency rescaling"})


def cheatsheet():
    return "kmyarn: theta_i / s^(2i/d); top frequency untouched, bottom scaled"
