# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parametric ReLU with a learnable negative slope."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_prelu"]


def geron_prelu(z, alpha=0.25, upstream=None):
    """
    Parametric ReLU: per-channel learnable negative slope.

    Formula: PReLU(z) = z if z>=0 else alpha_j*z, alpha_j learned

    Unlike leaky ReLU, alpha is a PARAMETER, so the gradient with respect
    to it is what makes the unit learn how leaky to be:
    dL/dalpha_j = sum over the negative pre-activations of channel j of
    (upstream * z). That sum is returned, so the caller can hand it to
    any optimiser. Channels are the last axis of ``z``; a scalar alpha is
    shared by all of them.

    The derivative at z = 0 is taken to be 1 (the positive branch);
    PReLU is not differentiable there.

    Parameters
    ----------
    z : array-like
        Pre-activations, channels on the last axis.
    alpha : float or array-like, default 0.25
        Negative slope, scalar or one per channel.
    upstream : array-like, optional
        dL/da from the next layer; default ones (so the reported
        gradients are the local ones).

    Returns
    -------
    result : RichResult
        Keys: a, grad_z, grad_alpha, negative_fraction, estimate, n,
        method.

    Examples
    --------
    >>> r = geron_prelu([-2.0, 3.0], 0.25)
    >>> [float(v) for v in r["a"]]
    [-0.5, 3.0]
    >>> [float(v) for v in r["grad_z"]]
    [0.25, 1.0]
    >>> float(r["grad_alpha"])
    -2.0

    Per-channel slopes on a two-channel row:

    >>> r2 = geron_prelu([[-4.0, -4.0]], [0.5, 0.0])
    >>> [float(v) for v in r2["a"][0]]
    [-2.0, -0.0]

    References
    ----------
    Geron Ch 11
    """
    a = np.atleast_1d(np.asarray(z, dtype=float))
    if a.size == 0:
        raise ValueError("geron_prelu: z is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError("geron_prelu: z contains non-finite values")
    C = a.shape[-1]
    al = np.asarray(alpha, dtype=float)
    if al.ndim == 0:
        alv = np.full(C, float(al))
        shared = True
    else:
        alv = al.ravel().astype(float)
        shared = False
        if alv.size != C:
            raise ValueError(f"geron_prelu: alpha has {alv.size} entries but z has {C} channels on its last axis")
    if not np.all(np.isfinite(alv)):
        raise ValueError("geron_prelu: alpha contains non-finite values")

    up = np.ones_like(a) if upstream is None else np.asarray(upstream, dtype=float)
    if up.shape != a.shape:
        raise ValueError(f"geron_prelu: upstream has shape {up.shape} but z has shape {a.shape}")

    neg = a < 0
    out = np.where(neg, alv * a, a)
    grad_z = np.where(neg, np.broadcast_to(alv, a.shape), 1.0) * up
    contrib = np.where(neg, a * up, 0.0)
    axes = tuple(range(a.ndim - 1))
    grad_alpha = contrib.sum(axis=axes) if a.ndim > 1 else contrib
    grad_alpha = np.asarray(grad_alpha, dtype=float).reshape(C)
    if shared:
        grad_alpha = float(grad_alpha.sum())

    return RichResult(
        title="PReLU",
        summary_lines=[("Channels", int(C)), ("Negative fraction", float(np.mean(neg)))],
        interpretation="alpha is learned; a channel that never goes negative gets zero gradient on its alpha.",
        payload={
            "a": out,
            "output": out,
            "grad_z": grad_z,
            "grad_alpha": grad_alpha,
            "alpha": alv if not shared else float(alv[0]),
            "negative_fraction": float(np.mean(neg)),
            "estimate": out,
            "n": int(a.size),
            "method": "PReLU forward with gradients w.r.t. z and alpha",
        },
    )


def cheatsheet():
    return "hmprel: Parametric ReLU with per-channel learnable slope"
