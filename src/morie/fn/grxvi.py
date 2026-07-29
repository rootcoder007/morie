# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Glorot (Xavier) initialization for a layer with given fan-in and fan-out."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_glorot_xavier_init"]

_METHOD = "Glorot (Xavier) initialization"


def _lcg(count, seed):
    """Deterministic uniforms in (0, 1): s = (1664525 s + 1013904223) mod 2^32."""
    s = int(seed) % 2**32
    out = np.empty(count)
    for i in range(count):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (s + 0.5) / 2**32
    return out


def geron_glorot_xavier_init(fan_in, fan_out, distribution="normal", seed=42):
    r"""Draw weights with the variance that keeps signal scale constant.

    .. math::
        \mathrm{Var}(W) = \frac{2}{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}

    realised either as :math:`\mathcal{N}(0, \sigma^2)` or as
    :math:`\mathcal{U}(-r, r)` with
    :math:`r = \sqrt{6/(\text{fan}_{in}+\text{fan}_{out})}` -- because a
    uniform on :math:`[-r, r]` has variance :math:`r^2/3`, and
    :math:`6/3 = 2` makes the two match exactly.

    The forward pass wants :math:`1/\text{fan}_{in}` to hold activation
    variance; the backward pass wants :math:`1/\text{fan}_{out}` to hold
    gradient variance.  Both cannot hold unless the layer is square, so
    Glorot takes the harmonic compromise.  The *achieved* sample variance
    over deterministic LCG draws is reported next to the target: with
    small layers the sample variance is legitimately several percent off,
    and quietly asserting otherwise would be a lie.

    Parameters
    ----------
    fan_in, fan_out : int
        Positive layer widths.
    distribution : {"normal", "uniform"}, optional
    seed : int, optional
        LCG seed.

    Returns
    -------
    RichResult
        Payload keys ``weights`` (fan_in x fan_out), ``target_variance``,
        ``achieved_variance``, ``scale`` (sigma or the uniform limit),
        ``estimate``, ``n``, ``method``.

    NOTE the normal branch fills the whole cosine block then the
    whole sine block of the Box-Muller pairs, where the sibling
    initialisers (grhei, grgrp, grdpmf) interleave the pairs; the
    distributions are identical, the draw ORDER is not, so streams
    are not interchangeable across the two layouts.

    References
    ----------
    Géron Ch 11, Eq 11-1 (Glorot initialization).

    Examples
    --------
    Target variance for a 4-in, 6-out layer is ``2/10 = 0.2``, and the
    uniform limit is ``sqrt(0.6) = 0.774597``:

    >>> r = geron_glorot_xavier_init(4, 6, distribution="uniform")
    >>> r["target_variance"]
    0.2
    >>> round(r["scale"], 6)
    0.774597

    Over enough draws the achieved variance lands near the target:

    >>> big = geron_glorot_xavier_init(100, 100, distribution="normal")
    >>> big["target_variance"]
    0.01
    >>> abs(big["achieved_variance"] - 0.01) < 0.001
    True
    """
    fan_in = int(fan_in)
    fan_out = int(fan_out)
    if fan_in < 1 or fan_out < 1:
        raise ValueError(f"fan_in and fan_out must be positive, got {fan_in} and {fan_out}.")
    target = 2.0 / (fan_in + fan_out)
    n = fan_in * fan_out
    u = _lcg(n if distribution == "uniform" else 2 * ((n + 1) // 2), seed)

    if distribution == "uniform":
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        W = (2.0 * u - 1.0) * limit
        scale = limit
    elif distribution == "normal":
        sigma = math.sqrt(target)
        # Box-Muller on the LCG uniforms: exact normals, no rejection loop.
        u1 = np.clip(u[0::2], 1e-12, 1.0)
        u2 = u[1::2]
        z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z2 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2 * np.pi * u2)
        W = np.concatenate([z, z2])[:n] * sigma
        scale = sigma
    else:
        raise ValueError(f"distribution must be 'normal' or 'uniform', got {distribution!r}.")

    W = W.reshape(fan_in, fan_out)
    achieved = float(np.var(W))

    return RichResult(
        title="Glorot initialization",
        summary_lines=[("Target variance", target), ("Achieved variance", achieved),
                       ("Distribution", distribution)],
        payload={
            "weights": W.tolist(),
            "target_variance": target,
            "achieved_variance": achieved,
            "scale": float(scale),
            "distribution": distribution,
            "estimate": W.tolist(),
            "n": int(n),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grxvi: Var(W) = 2/(fan_in+fan_out); uniform limit sqrt(6/(in+out)); achieved variance reported"
