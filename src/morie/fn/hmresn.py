# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ResNet residual block with a skip connection."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_resnet"]


def geron_resnet(x, F, projection=None):
    """
    ResNet: residual block with skip connection.

    Formula: y = F(x) + x

    The block learns the RESIDUAL, so an untrained or useless F leaves
    the identity behind rather than a random scramble -- a deep stack
    starts life as a working shallow network and only adds detours where
    they help. The gradient inherits the same structure,
    dy/dx = I + dF/dx, so the identity term keeps a path open even when
    every dF/dx is tiny; that is why depth stopped being fatal.

    ``F`` must return the shape it was given. When it cannot (a stride or
    a channel change), pass ``projection`` -- the 1x1-convolution role --
    to reshape the skip path.

    Parameters
    ----------
    x : array-like
        Block input.
    F : callable
        ``F(x) -> array`` of the same shape as ``x`` (or as
        ``projection(x)``).
    projection : callable, optional
        ``projection(x) -> array`` matching F's output shape.

    Returns
    -------
    result : RichResult
        Keys: y, residual, skip, residual_fraction, estimate, n, method.

    Examples
    --------
    A dead residual branch leaves the input untouched:

    >>> r = geron_resnet([1.0, 2.0], lambda a: np.zeros_like(a))
    >>> [float(v) for v in r["y"]]
    [1.0, 2.0]
    >>> float(r["residual_fraction"])
    0.0

    An identity branch doubles it:

    >>> [float(v) for v in geron_resnet([1.0, 2.0], lambda a: a)["y"]]
    [2.0, 4.0]

    A shape change without a projection is an error, not a broadcast:

    >>> geron_resnet([1.0, 2.0], lambda a: np.zeros(3))
    Traceback (most recent call last):
        ...
    ValueError: geron_resnet: F returned shape (3,) but the skip path has shape (2,)

    References
    ----------
    Geron Ch 12
    """
    if not callable(F):
        raise ValueError("geron_resnet: F must be callable")
    a = np.atleast_1d(np.asarray(x, dtype=float))
    if a.size == 0:
        raise ValueError("geron_resnet: x is empty")
    skip = a
    if projection is not None:
        if not callable(projection):
            raise ValueError("geron_resnet: projection must be callable")
        skip = np.asarray(projection(a), dtype=float)
    out = np.asarray(F(a), dtype=float)
    if out.shape != skip.shape:
        raise ValueError(f"geron_resnet: F returned shape {out.shape} but the skip path has shape {skip.shape}")
    if not np.all(np.isfinite(out)) or not np.all(np.isfinite(skip)):
        raise ValueError("geron_resnet: the block produced non-finite values")

    y = out + skip
    ns, nr = float(np.linalg.norm(skip)), float(np.linalg.norm(out))
    frac = nr / (nr + ns) if (nr + ns) > 0 else 0.0
    return RichResult(
        title="Residual block",
        summary_lines=[("||skip||", ns), ("||F(x)||", nr), ("Residual fraction", frac)],
        interpretation="dy/dx = I + dF/dx, so the identity keeps a gradient path open through arbitrary depth.",
        payload={
            "y": y,
            "output": y,
            "residual": out,
            "skip": skip,
            "residual_fraction": frac,
            "estimate": y,
            "n": int(a.size),
            "method": "Residual block y = F(x) + x (optionally projected skip)",
        },
    )


def cheatsheet():
    return "hmresn: ResNet residual block y = F(x) + x"
