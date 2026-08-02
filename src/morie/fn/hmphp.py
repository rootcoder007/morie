# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Peephole LSTM cell: gates also look at the cell state."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_peephole_lstm"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def geron_peephole_lstm(x_t, h_prev, c_prev, weights):
    """
    Peephole LSTM: gates also look at the cell state.

    Formula: i_t = sigma(W_x x_t + W_h h_{t-1} + W_c c_{t-1} + b)

    In a plain LSTM the gates see only the input and the previous OUTPUT
    h, which is the cell state already squashed and masked by the output
    gate -- so when o is closed the gates are blind to what the cell
    holds. The peephole connections give the input and forget gates a
    direct view of c_{t-1}, and the output gate a view of the freshly
    computed c_t, which is what lets the cell learn precise timings.

    ``weights`` is a mapping with ``W_x`` (4H, n), ``W_h`` (4H, H), ``b``
    (4H,) stacked in the order [i, f, g, o], plus the peephole vectors
    ``p_i``, ``p_f``, ``p_o`` of length H (Hadamard, not matrices).

    Parameters
    ----------
    x_t : array-like, shape (n,)
    h_prev, c_prev : array-like, shape (H,)
    weights : mapping
        As described above; missing peepholes default to zeros, which
        recovers the plain LSTM.

    Returns
    -------
    result : RichResult
        Keys: h, c, i, f, g, o, estimate, n, method.

    Examples
    --------
    All weights zero except the candidate row, so i = f = o = 1/2,
    g = tanh(1), c = 0.5*tanh(1) and h = 0.5*tanh(c):

    >>> W = {"W_x": [[0.0], [0.0], [1.0], [0.0]], "W_h": [[0.0]] * 4, "b": [0.0] * 4}
    >>> r = geron_peephole_lstm([1.0], [0.0], [0.0], W)
    >>> round(float(r["g"][0]), 6), round(float(r["c"][0]), 6)
    (0.761594, 0.380797)
    >>> round(float(r["h"][0]), 6)
    0.1817

    A strong forget peephole opens the gate once the cell is charged:

    >>> W2 = dict(W, p_f=[10.0])
    >>> round(float(geron_peephole_lstm([1.0], [0.0], [1.0], W2)["f"][0]), 7)
    0.9999546

    References
    ----------
    Geron Ch 13
    """
    if not hasattr(weights, "get"):
        raise ValueError("geron_peephole_lstm: weights must be a mapping with W_x, W_h and b")
    x = np.atleast_1d(np.asarray(x_t, dtype=float)).ravel()
    h = np.atleast_1d(np.asarray(h_prev, dtype=float)).ravel()
    c = np.atleast_1d(np.asarray(c_prev, dtype=float)).ravel()
    H = h.size
    if c.size != H:
        raise ValueError(f"geron_peephole_lstm: h_prev has {H} units but c_prev has {c.size}")
    for key in ("W_x", "W_h", "b"):
        if weights.get(key) is None:
            raise ValueError(f"geron_peephole_lstm: weights is missing {key!r}")
    Wx = np.atleast_2d(np.asarray(weights["W_x"], dtype=float))
    Wh = np.atleast_2d(np.asarray(weights["W_h"], dtype=float))
    b = np.atleast_1d(np.asarray(weights["b"], dtype=float)).ravel()
    if Wx.shape != (4 * H, x.size):
        raise ValueError(f"geron_peephole_lstm: W_x has shape {Wx.shape}, expected {(4 * H, x.size)}")
    if Wh.shape != (4 * H, H):
        raise ValueError(f"geron_peephole_lstm: W_h has shape {Wh.shape}, expected {(4 * H, H)}")
    if b.size != 4 * H:
        raise ValueError(f"geron_peephole_lstm: b has {b.size} entries, expected {4 * H}")

    def _peep(name):
        v = weights.get(name)
        if v is None:
            return np.zeros(H)
        a = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
        if a.size != H:
            raise ValueError(f"geron_peephole_lstm: {name} has {a.size} entries but there are {H} units")
        return a

    p_i, p_f, p_o = _peep("p_i"), _peep("p_f"), _peep("p_o")
    for name, arr in (("x_t", x), ("h_prev", h), ("c_prev", c), ("W_x", Wx), ("W_h", Wh), ("b", b)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"geron_peephole_lstm: {name} contains non-finite values")

    z = Wx @ x + Wh @ h + b
    i = _sigmoid(z[0:H] + p_i * c)
    f = _sigmoid(z[H : 2 * H] + p_f * c)
    g = np.tanh(z[2 * H : 3 * H])
    c_new = f * c + i * g
    o = _sigmoid(z[3 * H : 4 * H] + p_o * c_new)  # peephole on the NEW cell state
    h_new = o * np.tanh(c_new)

    return RichResult(
        title="Peephole LSTM cell",
        summary_lines=[("Units", int(H)), ("Mean forget gate", float(np.mean(f)))],
        interpretation="i and f peep at c_{t-1}, o at c_t; that timing information is invisible to a plain LSTM.",
        payload={
            "h": h_new,
            "c": c_new,
            "i": i,
            "f": f,
            "g": g,
            "o": o,
            "estimate": h_new,
            "n": int(H),
            "method": "Peephole LSTM cell forward step",
        },
    )


def cheatsheet():
    return "hmphp: Peephole LSTM cell, gates see the cell state"
