# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 4.1: the SERIES adapter update."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch4_series_adapter"]


def _relu(a):
    return np.maximum(a, 0.0)


def _adapter_core(H_o, H_in, W_down, W_up, f):
    """Shared bottleneck adapter: H_o + f(H_in W_down) W_up.

    Series (Eq 4.1) passes H_in = H_o; parallel (Eq 4.2) passes the
    layer INPUT H_i. km055 imports this so the two adapters differ only
    where the book says they differ.
    """
    H_o = np.atleast_2d(np.asarray(H_o, dtype=float))
    H_in = np.atleast_2d(np.asarray(H_in, dtype=float))
    Wd = np.atleast_2d(np.asarray(W_down, dtype=float))
    Wu = np.atleast_2d(np.asarray(W_up, dtype=float))
    if H_in.shape[1] != Wd.shape[0]:
        raise ValueError(
            f"W_down has {Wd.shape[0]} rows but the adapter input has "
            f"width {H_in.shape[1]}.")
    if Wd.shape[1] != Wu.shape[0]:
        raise ValueError(
            f"the bottleneck disagrees: W_down is {Wd.shape}, W_up is "
            f"{Wu.shape}.")
    if Wu.shape[1] != H_o.shape[1]:
        raise ValueError(
            f"W_up returns width {Wu.shape[1]} but H_o has width "
            f"{H_o.shape[1]}; the residual could not be added.")
    if H_in.shape[0] != H_o.shape[0]:
        raise ValueError("the adapter input and H_o must have the same "
                         "number of rows (positions).")
    if not callable(f):
        raise ValueError("f must be a callable activation.")
    delta = f(H_in @ Wd) @ Wu
    return H_o + delta, delta, Wd.shape[1]


def kamath_ch4_series_adapter(H_o, W_down, W_up, f=None):
    """H_o <- H_o + f(H_o W_down) W_up.

    The series adapter reads its OWN output: the bottleneck sees H_o.
    ``f`` defaults to ReLU. The added term is a residual, so with a
    zero W_up the layer is unchanged -- the initialisation adapters
    rely on.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, Eq 4.1, printed
    p. 147.

    Examples
    --------
    >>> out = kamath_ch4_series_adapter([[1.0, 2.0]], [[1.0], [0.0]],
    ...                                 [[1.0, 1.0]])
    >>> out["output"]
    [[2.0, 3.0]]
    >>> out["bottleneck_rank"]
    1
    """
    out, delta, r = _adapter_core(H_o, H_o, W_down, W_up,
                                  _relu if f is None else f)
    return RichResult(payload={
        "output": [[float(v) for v in row] for row in out],
        "delta": [[float(v) for v in row] for row in delta],
        "bottleneck_rank": int(r), "estimate": float(out[0, 0]),
        "n": int(out.shape[0]),
        "method": "series adapter (Kamath Eq 4.1)"})


def cheatsheet():
    return "km054: H_o + f(H_o W_down) W_up, series adapter"
