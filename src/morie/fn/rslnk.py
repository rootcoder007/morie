# morie.fn -- function file (rootcoder007/morie)
"""Residual / skip connection."""

from __future__ import annotations

from collections.abc import Callable

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["residual_connection"]


def residual_connection(x, f: Callable[[np.ndarray], np.ndarray] | None = None):
    r"""Residual (identity-shortcut) connection.

    .. math::

        y = \\mathcal{F}(x) + x

    where :math:`\\mathcal{F}` is an arbitrary callable layer.  If
    ``f`` is ``None``, :math:`\\mathcal{F}` defaults to the identity
    (so :math:`y = 2x`).

    Parameters
    ----------
    x : array-like
        Input.
    f : callable or array-like, optional
        The residual branch. Either the layer itself -- any callable taking
        ``x`` and returning a shape-compatible array -- or an already-computed
        ``F(x)``.

        The array form is accepted because a caller assembling a block by
        hand usually has ``F(x)`` in a variable already, and passing it used
        to fail several frames in with ``'numpy.ndarray' object is not
        callable``, naming neither the argument nor what it wanted. That
        signature appears four times in the audit's own red list.

    Returns
    -------
    result : RichResult
        Keys: ``y`` / ``estimate``, ``Fx``.

    References
    ----------
    He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual
    learning for image recognition. *CVPR*.
    """
    x = np.asarray(x, dtype=float)
    if f is None:
        Fx = x
    elif callable(f):
        Fx = np.asarray(f(x), dtype=float)
    else:
        # Already-computed F(x).
        Fx = np.asarray(f, dtype=float)
    if Fx.shape != x.shape:
        raise ValueError(f"Residual branch shape {Fx.shape} != identity shape {x.shape}.")
    y = Fx + x
    return RichResult(
        title="Residual connection",
        summary_lines=[("shape", x.shape)],
        payload={
            "y": y,
            "estimate": y,
            "Fx": Fx,
            "method": "Residual identity shortcut",
        },
    )


# CANONICAL TEST
# residual_connection([1,2,3], f=lambda x: x*2).y -> [3,6,9]


def cheatsheet():
    return "rslnk: Residual y = F(x) + x"
