# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burkov Ch 3: one step of the Elman RNN."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["burkov_elman_rnn"]


def burkov_elman_rnn(x_t, h_prev, Wh, Wx, Wy, bh, by):
    """h_t = tanh(Wh h_prev + Wx x_t + bh); y_t = Wy h_t + by.

    References: Burkov LM (2025), Ch 3, Elman RNN (Elman 1990).

    Examples
    --------
    >>> out = burkov_elman_rnn([1.0], [0.0], [[0.0]], [[0.0]],
    ...                        [[1.0]], [0.0], [0.0])
    >>> out["h"]
    [0.0]
    """
    x = np.atleast_1d(np.asarray(x_t, dtype=float))
    h0 = np.atleast_1d(np.asarray(h_prev, dtype=float))
    Wh = np.atleast_2d(np.asarray(Wh, dtype=float))
    Wx = np.atleast_2d(np.asarray(Wx, dtype=float))
    Wy = np.atleast_2d(np.asarray(Wy, dtype=float))
    bh = np.atleast_1d(np.asarray(bh, dtype=float))
    by = np.atleast_1d(np.asarray(by, dtype=float))
    if Wh.shape != (len(h0), len(h0)):
        raise ValueError(
            f"Wh must be {len(h0)} x {len(h0)}; got {Wh.shape}.")
    if Wx.shape != (len(h0), len(x)):
        raise ValueError(
            f"Wx must be {len(h0)} x {len(x)}; got {Wx.shape}.")
    if Wy.shape[1] != len(h0):
        raise ValueError(
            f"Wy must have {len(h0)} columns; got {Wy.shape}.")
    if len(bh) != len(h0) or len(by) != Wy.shape[0]:
        raise ValueError("bias lengths must match Wh rows and Wy rows.")
    h = np.tanh(Wh @ h0 + Wx @ x + bh)
    y = Wy @ h + by
    return RichResult(payload={
        "h": [float(v) for v in h], "y": [float(v) for v in y],
        "estimate": float(y[0]), "n": len(h),
        "method": "Elman RNN step (Burkov Ch 3)"})


def cheatsheet():
    return "bkelm: Elman RNN recurrence h_t = tanh(Wh h + Wx x + bh) (Burkov Ch 3)"


# compact alias per ledger/NAMING.md
burkovelmanrnn = burkov_elman_rnn
