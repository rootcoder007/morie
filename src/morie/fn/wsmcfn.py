# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Characteristic function phi_X(t) = E[e^{itX}]."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_char_fn"]


def wasserman_char_fn(x, t):
    """
    Empirical characteristic function.

    Formula: phi_X(t) = E[e^{itX}] = E[cos(tX)] + i E[sin(tX)],
    estimated on the sample. Real and imaginary parts are returned
    separately (payloads stay real-valued); |phi(t)| <= 1 always and
    phi(0) = 1 exactly.

    Parameters
    ----------
    x : array-like
        Sample (non-empty).
    t : array-like
        Evaluation point(s).

    Returns
    -------
    result : dict
        Keys: estimate (|phi| at the first t), real, imag, modulus,
        t, n, method.

    References
    ----------
    Wasserman (2004), Ch 3.

    Examples
    --------
    >>> out = wasserman_char_fn([1.0, 2.0, 3.0], 0.0)
    >>> out["estimate"]
    1.0
    >>> out["imag"]
    [0.0]
    >>> import math
    >>> out = wasserman_char_fn([math.pi], 1.0)
    >>> abs(out["real"][0] - (-1.0)) < 1e-15
    True
    >>> all(m <= 1.0 + 1e-12 for m in wasserman_char_fn([0.3, -2.0, 5.5], [0.5, 1.5, 9.0])["modulus"])
    True
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if x.size == 0:
        raise ValueError("the characteristic function of an empty sample is undefined.")
    re = [float(np.mean(np.cos(ti * x))) for ti in t]
    im = [float(np.mean(np.sin(ti * x))) for ti in t]
    mod = [float(np.hypot(a, b)) for a, b in zip(re, im)]
    return RichResult(payload={
        "estimate": mod[0], "real": re, "imag": im, "modulus": mod,
        "t": [float(v) for v in t], "n": int(x.size),
        "method": "empirical phi(t) = mean cos(tX) + i mean sin(tX)"})


def cheatsheet():
    return "wsmcfn: phi(t) = E[cos tX] + iE[sin tX]; |phi| <= 1, phi(0) = 1"
