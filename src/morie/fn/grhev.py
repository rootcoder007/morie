# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Heaviside step activation function."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_heaviside_step"]

_METHOD = "Heaviside step activation"


def geron_heaviside_step(z, threshold=0.0):
    r"""The threshold logic unit's activation.

    .. math::
        \mathrm{heaviside}(z) = \begin{cases}
        1 & z \ge 0\\ 0 & z < 0\end{cases}

    Note the closed inequality at zero: ``heaviside(0) = 1``, which is
    the convention Géron's TLU uses.  ``numpy.heaviside`` leaves that
    point as a free parameter, so the choice is made explicit here.

    Its derivative is zero wherever it is defined -- that is exactly why
    perceptrons are trained by Hebb's rule (:mod:`morie.fn.grhbb`) and
    not by gradient descent.

    Parameters
    ----------
    z : array-like or float
        Pre-activation.
    threshold : float, optional
        Step location, default 0.

    Returns
    -------
    RichResult
        Payload keys ``output``, ``fraction_active``, ``threshold``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 9, Heaviside step activation.

    Examples
    --------
    Zero fires, anything below it does not:

    >>> r = geron_heaviside_step([-2.0, -0.001, 0.0, 3.0])
    >>> r["output"]
    [0.0, 0.0, 1.0, 1.0]
    >>> r["fraction_active"]
    0.5

    A scalar comes back as a scalar:

    >>> geron_heaviside_step(-1.0)["output"]
    0.0
    """
    z_arr = np.asarray(z, dtype=float)
    if not np.all(np.isfinite(z_arr)):
        raise ValueError("z must be finite; the step is undefined at nan.")
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold}.")

    out = (z_arr >= threshold).astype(float)
    scalar = z_arr.ndim == 0

    return RichResult(
        title="Heaviside step",
        summary_lines=[("Active fraction", float(out.mean())), ("Threshold", threshold)],
        payload={
            "output": float(out) if scalar else out.tolist(),
            "fraction_active": float(out.mean()),
            "threshold": threshold,
            "estimate": float(out) if scalar else out.tolist(),
            "n": 1 if scalar else int(out.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grhev: heaviside(z) = 1 if z >= 0 else 0 (closed at zero)"
