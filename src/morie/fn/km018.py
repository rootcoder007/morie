# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 2.18: layer normalisation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch2_layer_norm"]


def kamath_ch2_layer_norm(h_i, mu=None, sigma=None, g=1.0, eps=1e-5):
    """h = g (h - mu) / sigma; mu and sigma default to the vector's
    own mean and standard deviation (the definition of LAYER norm);
    supplying them pins the book's notation. The normalised output has
    mean 0 and sd 1 before the gain, asserted in the tests.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 2, Eq 2.18, printed
    p. 37 (PDF-verified page map: printed = PDF - 27).

    Examples
    --------
    >>> out = kamath_ch2_layer_norm([1.0, 3.0])
    >>> out["normalised"]
    [-1.0, 1.0]
    """
    h = np.atleast_1d(np.asarray(h_i, dtype=float))
    if len(h) < 2 and mu is None:
        raise ValueError(
            "layer norm over a single element is 0/0; supply mu and "
            "sigma or a longer vector.")
    m = float(np.mean(h)) if mu is None else float(mu)
    s = float(np.std(h)) if sigma is None else float(sigma)
    if s <= 0:
        raise ValueError("sigma must be positive; a constant vector "
                         "cannot be layer-normalised.")
    gv = np.atleast_1d(np.asarray(g, dtype=float))
    out = gv * (h - m) / (s + eps * 0)
    normed = (h - m) / s
    return RichResult(payload={
        "output": [float(v) for v in out],
        "normalised": [float(v) for v in normed],
        "mu": m, "sigma": s, "estimate": float(out[0]), "n": len(h),
        "method": "Layer normalisation g(h - mu)/sigma "
                  "(Kamath Eq 2.18)"})


def cheatsheet():
    return "km018: layer norm, statistics computed or pinned"
