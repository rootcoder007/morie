# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Glorot (Xavier) initialization for sigmoid/tanh networks."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_glorot_init"]


def _lcg_uniform(size, seed):
    """Deterministic LCG stream on (0, 1) -- no RNG state, no dependencies."""
    s = int(seed) % 2**32
    out = np.empty(int(size))
    for i in range(int(size)):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (s + 0.5) / 2**32
    return out


def geron_glorot_init(fan_in, fan_out, seed=0, distribution="uniform"):
    """
    Glorot (Xavier) initialization for sigmoid/tanh networks.

    Formula: Var(W) = 2 / (fan_in + fan_out)

    The uniform form draws on ``[-limit, limit]`` with
    ``limit = sqrt(6/(fan_in+fan_out))``, whose variance is
    ``limit^2/3 = 2/(fan_in+fan_out)`` exactly -- the same target the
    normal form hits with ``sigma = sqrt(2/(fan_in+fan_out))``. Weights
    are drawn from a linear congruential stream so the matrix is
    reproducible without touching global RNG state.

    Parameters
    ----------
    fan_in, fan_out : int
        Incoming and outgoing connection counts (both >= 1).
    seed : int, default 0
        LCG seed.
    distribution : {"uniform", "normal"}, default "uniform"
        Which Glorot variant to draw.

    Returns
    -------
    result : RichResult
        Keys: W, limit, std, variance, estimate, n, method.

    Examples
    --------
    >>> r = geron_glorot_init(100, 100)
    >>> round(float(r["limit"]), 6)
    0.173205
    >>> round(float(r["variance"]), 6)
    0.01
    >>> r["W"].shape
    (100, 100)
    >>> bool(abs(r["W"]).max() <= r["limit"])
    True

    References
    ----------
    Géron Ch 11
    """
    fi, fo = int(fan_in), int(fan_out)
    if fi < 1 or fo < 1:
        raise ValueError(f"geron_glorot_init: fan_in and fan_out must be >= 1, got {fi} and {fo}")
    dist = str(distribution).lower()
    if dist not in ("uniform", "normal"):
        raise ValueError(f"geron_glorot_init: distribution must be 'uniform' or 'normal', got {distribution!r}")

    var = 2.0 / (fi + fo)
    std = float(np.sqrt(var))
    limit = float(np.sqrt(6.0 / (fi + fo)))
    u = _lcg_uniform(fi * fo, seed)
    if dist == "uniform":
        W = (2.0 * u - 1.0) * limit
    else:
        # Box-Muller on two independent LCG streams, truncated to fi*fo draws.
        u2 = _lcg_uniform(fi * fo, int(seed) + 7919)
        W = std * np.sqrt(-2.0 * np.log(u)) * np.cos(2.0 * np.pi * u2)
    W = W.reshape(fi, fo)

    return RichResult(
        title="Glorot (Xavier) initialization",
        summary_lines=[("fan_in", fi), ("fan_out", fo), ("Target variance", var), ("Uniform limit", limit)],
        interpretation=(
            "Glorot keeps the forward-signal and backward-gradient variances comparable across a "
            "sigmoid/tanh layer; use He initialization for ReLU instead."
        ),
        payload={
            "W": W,
            "limit": limit,
            "std": std,
            "variance": var,
            "fan_in": fi,
            "fan_out": fo,
            "distribution": dist,
            "estimate": var,
            "n": int(fi * fo),
            "method": f"Glorot {dist} initialization, Var(W) = 2/(fan_in + fan_out)",
        },
    )


def cheatsheet():
    return "hmxav: Glorot (Xavier) initialization for sigmoid/tanh networks"
