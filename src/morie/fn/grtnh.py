# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hyperbolic tangent activation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_tanh_activation"]

_METHOD = "Hyperbolic tangent activation"


def geron_tanh_activation(z):
    r"""tanh activation and its derivative.

    .. math::
        \tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}},
        \qquad \tanh'(z) = 1 - \tanh^2(z)

    The ratio-of-exponentials form overflows for ``|z| > 710``; the
    identity :math:`\tanh(z) = 2\sigma(2z) - 1` does not, and
    ``np.tanh`` implements exactly that saturating evaluation, so it is
    what gets called here.  Unlike the logistic, tanh is zero-centred,
    which is why Géron prefers it for hidden layers.

    Parameters
    ----------
    z : array-like
        Pre-activation(s), any shape. Must be finite.

    Returns
    -------
    RichResult
        Payload keys ``activation``, ``derivative``, ``saturated``
        (fraction with ``|tanh| > 0.99``), ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 9, Activation Functions (tanh).

    Examples
    --------
    >>> r = geron_tanh_activation([0.0, 1.0, -1.0])
    >>> [round(v, 6) for v in r["activation"]]
    [0.0, 0.761594, -0.761594]
    >>> round(r["derivative"][1], 6)
    0.419974

    Zero-centred, unlike the logistic:

    >>> round(sum(r["activation"]), 12)
    0.0
    """
    z = np.asarray(z, dtype=float)
    if z.size == 0:
        raise ValueError("z is empty; tanh needs at least one pre-activation.")
    if not np.all(np.isfinite(z)):
        raise ValueError("z contains non-finite values.")

    a = np.tanh(z)
    d = 1.0 - a * a
    est = float(a) if a.ndim == 0 else a.tolist()
    return RichResult(
        title="tanh activation",
        summary_lines=[("n", int(z.size)), ("saturated fraction", float(np.mean(np.abs(a) > 0.99)))],
        payload={
            "activation": est,
            "derivative": float(d) if d.ndim == 0 else d.tolist(),
            "saturated": float(np.mean(np.abs(a) > 0.99)),
            "estimate": est,
            "n": int(z.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grtnh: tanh(z) = (e^z - e^-z)/(e^z + e^-z); derivative 1 - tanh^2; zero-centred"
