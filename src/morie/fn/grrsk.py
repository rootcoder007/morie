# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ResNet residual skip connection: y = F(x) + x."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_resnet_skip"]

_METHOD = "Residual (skip) connection"


def geron_resnet_skip(x, Fx, projection=None):
    r"""Add the input back onto the block output.

    .. math::
        \mathbf{y} = F(\mathbf{x}; \theta) + \mathbf{x}

    The identity term is what makes very deep stacks trainable: the
    gradient of ``y`` w.r.t. ``x`` is :math:`\partial F/\partial x + I`,
    so it can never vanish through the shortcut, however small the
    learned Jacobian is.  Initialised near zero, the block starts as the
    identity map and *learns the difference* -- hence "residual".  When
    the block changes shape, a projection matrix is required; silently
    broadcasting would be a bug, so a shape mismatch without
    ``projection`` raises.

    Parameters
    ----------
    x : array-like
        Block input.
    Fx : array-like
        Block output ``F(x)``.
    projection : array-like, optional
        ``(d_in, d_out)`` matrix applied to ``x`` when the shapes differ
        (ResNet's 1x1 "option B" shortcut).

    Returns
    -------
    RichResult
        Payload keys ``output``, ``shortcut``, ``residual_norm``,
        ``shortcut_norm``, ``residual_fraction``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 12, ResNet / Skip Connections section.

    Examples
    --------
    >>> r = geron_resnet_skip([1.0, 2.0], [0.5, -1.0])
    >>> r["output"]
    [1.5, 1.0]

    A zero residual leaves the identity behind, which is exactly the
    initialisation ResNet relies on:

    >>> geron_resnet_skip([1.0, 2.0], [0.0, 0.0])["output"]
    [1.0, 2.0]

    Widening needs an explicit projection:

    >>> geron_resnet_skip([1.0], [0.0, 0.0])
    Traceback (most recent call last):
        ...
    ValueError: F(x) has shape (2,) but x has shape (1,); supply projection= for a shape-changing block.
    """
    xa = np.asarray(x, dtype=float)
    fa = np.asarray(Fx, dtype=float)
    if xa.size == 0:
        raise ValueError("x is empty.")
    if not np.all(np.isfinite(xa)) or not np.all(np.isfinite(fa)):
        raise ValueError("x and F(x) must be finite.")

    short = xa
    if fa.shape != xa.shape:
        if projection is None:
            raise ValueError(
                f"F(x) has shape {fa.shape} but x has shape {xa.shape}; "
                "supply projection= for a shape-changing block."
            )
        P = np.atleast_2d(np.asarray(projection, dtype=float))
        if P.ndim != 2 or P.shape[0] != xa.shape[-1]:
            raise ValueError(
                f"projection must be (d_in={xa.shape[-1]}, d_out), got shape {P.shape}."
            )
        short = xa @ P
        if short.shape != fa.shape:
            raise ValueError(
                f"projected shortcut has shape {short.shape} but F(x) has {fa.shape}."
            )

    out = fa + short
    fn = float(np.linalg.norm(fa))
    sn = float(np.linalg.norm(short))
    return RichResult(
        title="Residual skip connection",
        summary_lines=[("||F(x)||", fn), ("||shortcut||", sn)],
        payload={
            "output": out.tolist(),
            "shortcut": short.tolist(),
            "residual_norm": fn,
            "shortcut_norm": sn,
            "residual_fraction": float(fn / (fn + sn)) if (fn + sn) > 0 else 0.0,
            "estimate": out.tolist(),
            "n": int(out.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grrsk: y = F(x) + x; d y/d x = dF/dx + I never vanishes; shape change needs projection="
