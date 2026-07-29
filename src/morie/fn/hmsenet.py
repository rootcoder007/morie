# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Squeeze-and-Excitation (SENet) block for channel recalibration."""

import numpy as np

from ._richresult import RichResult
from .hmsigm import geron_sigmoid

__all__ = ["geron_senet"]


def _lcg(shape, seed, scale=0.5):
    n = int(np.prod(shape))
    s = int(seed) % 2**32
    out = np.empty(n)
    for i in range(n):
        s = (1664525 * s + 1013904223) % 2**32
        out[i] = (2.0 * ((s + 0.5) / 2**32) - 1.0) * scale
    return out.reshape(shape)


def geron_senet(x, r=16, W1=None, W2=None, seed=0):
    """
    Squeeze-and-Excitation (SENet) block for channel recalibration.

    Formula: s = sigmoid(W2 ReLU(W1 GAP(x))); y = s * x

    The three stages are executed literally: *squeeze* is global average
    pooling over the spatial axes, *excitation* is the bottleneck MLP
    ``C -> C/r -> C`` with ReLU then sigmoid (delegated to
    :func:`morie.fn.hmsigm.geron_sigmoid`), and *scale* multiplies each
    channel of the input by its gate. The block is shape-preserving and
    adds ``2*C^2/r`` parameters.

    Parameters
    ----------
    x : array-like
        Feature map (H, W, C), or (C,) for an already-pooled descriptor.
    r : int, default 16
        Reduction ratio; must divide C and leave at least one hidden unit.
    W1, W2 : array-like, optional
        Excitation weights of shape (C, C/r) and (C/r, C). Defaults are
        deterministic LCG draws.
    seed : int, default 0
        LCG seed used when W1/W2 are not supplied.

    Returns
    -------
    result : RichResult
        Keys: y, s, z, n_params, estimate, n, method.

    Examples
    --------
    Gates are probabilities, one per channel, and the output keeps the
    input's shape:

    >>> x = [[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]]
    >>> r = geron_senet(x, r=2)
    >>> r["y"].shape, r["s"].shape
    ((2, 2, 2), (2,))
    >>> bool((r["s"] > 0).all() and (r["s"] < 1).all())
    True
    >>> [float(v) for v in r["z"]]
    [2.5, 25.0]

    With identity excitation weights the gate is just sigmoid of the
    squeezed descriptor:

    >>> I = [[1.0, 0.0], [0.0, 1.0]]
    >>> r2 = geron_senet([1.0, 0.0], r=1, W1=I, W2=I)
    >>> [round(float(v), 6) for v in r2["s"]]
    [0.731059, 0.5]
    >>> int(r2["n_params"])
    8

    References
    ----------
    Géron Ch 12
    """
    X = np.asarray(x, dtype=float)
    if X.size == 0:
        raise ValueError("geron_senet: x is empty")
    if not np.all(np.isfinite(X)):
        raise ValueError("geron_senet: x contains non-finite values")
    if X.ndim == 1:
        z = X.copy()
    elif X.ndim == 3:
        z = X.mean(axis=(0, 1))
    else:
        raise ValueError(f"geron_senet: x must be (H, W, C) or (C,), got shape {X.shape}")
    C = z.size
    red = int(r)
    if red < 1:
        raise ValueError(f"geron_senet: reduction ratio r must be >= 1, got {red}")
    if C % red:
        raise ValueError(f"geron_senet: reduction ratio r={red} does not divide C={C}")
    hidden = C // red
    if hidden < 1:
        raise ValueError(f"geron_senet: r={red} leaves {hidden} hidden units; the bottleneck would vanish")

    A = _lcg((C, hidden), int(seed) + 1) if W1 is None else np.asarray(W1, dtype=float)
    B = _lcg((hidden, C), int(seed) + 2) if W2 is None else np.asarray(W2, dtype=float)
    if A.shape != (C, hidden):
        raise ValueError(f"geron_senet: W1 must have shape {(C, hidden)}, got {A.shape}")
    if B.shape != (hidden, C):
        raise ValueError(f"geron_senet: W2 must have shape {(hidden, C)}, got {B.shape}")

    h = np.maximum(z @ A, 0.0)
    s = np.asarray(geron_sigmoid(h @ B)["a"], dtype=float)
    y = X * s

    return RichResult(
        title="Squeeze-and-Excitation block",
        summary_lines=[
            ("Channels", int(C)),
            ("Bottleneck", int(hidden)),
            ("Min gate", float(np.min(s))),
            ("Max gate", float(np.max(s))),
        ],
        interpretation=(
            "SE gives the network a global, per-channel attention: cheap (2*C^2/r parameters) because "
            "the spatial map is squeezed to one number per channel first."
        ),
        payload={
            "y": y,
            "s": s,
            "gate": s,
            "z": z,
            "hidden": int(hidden),
            "n_params": int(2 * C * hidden),
            "estimate": float(np.mean(s)),
            "n": int(C),
            "method": "Squeeze (GAP) -> excite (bottleneck MLP + sigmoid) -> scale",
        },
    )


def cheatsheet():
    return "hmsenet: Squeeze-and-Excitation (SENet) block for channel recalibration"
