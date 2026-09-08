# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LSTM cell: input/forget/output gates + cell state."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_lstm"]

_METHOD = "LSTM cell forward step"

_KEYS = ("W_i", "U_i", "b_i", "W_f", "U_f", "b_f", "W_o", "U_o", "b_o", "W_g", "U_g", "b_g")


def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def geron_lstm(x_t, h_prev, c_prev, weights):
    """
    LSTM cell: input/forget/output gates + cell state.

    Formula: i_t = sigma(...); f_t = sigma(...); o_t = sigma(...); c_t = f_t*c_{t-1} + i_t*g_t

    One LSTM time step:

    ``i_t = sigma(W_i x + U_i h + b_i)``  (input gate)
    ``f_t = sigma(W_f x + U_f h + b_f)``  (forget gate)
    ``o_t = sigma(W_o x + U_o h + b_o)``  (output gate)
    ``g_t = tanh(W_g x + U_g h + b_g)``   (candidate)
    ``c_t = f_t * c_{t-1} + i_t * g_t``
    ``h_t = o_t * tanh(c_t)``

    The long-term path ``c`` is touched only by elementwise
    multiplication and addition -- no repeated matrix product -- which is
    why the gradient along it decays like a product of forget gates
    rather than a product of Jacobians.  That is the whole reason the
    cell state exists, so ``f_t`` is reported: a forget gate near 1
    keeps the memory alive, near 0 wipes it in one step.

    Parameters
    ----------
    x_t : array-like, shape (n_in,)
        Input at this step.
    h_prev : array-like, shape (n_units,)
        Previous short-term state.
    c_prev : array-like, shape (n_units,)
        Previous long-term state.
    weights : mapping
        W_i, U_i, b_i, W_f, U_f, b_f, W_o, U_o, b_o, W_g, U_g, b_g with
        W_* of shape (n_units, n_in), U_* of shape (n_units, n_units)
        and b_* of shape (n_units,).

    Returns
    -------
    result : RichResult
        Keys: h_t, c_t, i_t, f_t, o_t, g_t, estimate, n, method.

    Examples
    --------
    All-zero weights: every gate is 0.5, the candidate is 0, so
    ``c_t = 0.5*c_prev`` and ``h_t = 0.5*tanh(c_t)``.  With
    ``c_prev = [2, -2]`` that is ``c_t = [1, -1]``:

    >>> import numpy as np
    >>> Z = {k: (np.zeros((2, 2)) if k[0] in "WU" else np.zeros(2)) for k in
    ...      ("W_i", "U_i", "b_i", "W_f", "U_f", "b_f",
    ...       "W_o", "U_o", "b_o", "W_g", "U_g", "b_g")}
    >>> r = geron_lstm([0.0, 0.0], [0.0, 0.0], [2.0, -2.0], Z)
    >>> [float(v) for v in r["c_t"]]
    [1.0, -1.0]
    >>> [round(float(v), 9) for v in r["h_t"]]
    [0.380797078, -0.380797078]

    A forget gate driven to 1 and an input gate to 0 carry the memory
    through untouched:

    >>> W = dict(Z); W["b_f"] = np.array([40.0, 40.0]); W["b_i"] = np.array([-40.0, -40.0])
    >>> k = geron_lstm([0.0, 0.0], [0.0, 0.0], [2.0, -2.0], W)
    >>> [round(float(v), 9) for v in k["c_t"]]
    [2.0, -2.0]

    A forget gate driven to 0 wipes it:

    >>> W2 = dict(Z); W2["b_f"] = np.array([-40.0, -40.0]); W2["b_i"] = np.array([-40.0, -40.0])
    >>> [round(float(v), 9) for v in geron_lstm([0.0, 0.0], [0.0, 0.0], [2.0, -2.0], W2)["c_t"]]
    [0.0, -0.0]

    References
    ----------
    Géron Ch 13
    """
    if not hasattr(weights, "__getitem__"):
        raise ValueError("geron_lstm: weights must be a mapping with keys " + ", ".join(_KEYS))
    x = np.atleast_1d(np.asarray(x_t, dtype=float)).ravel()
    h = np.atleast_1d(np.asarray(h_prev, dtype=float)).ravel()
    c = np.atleast_1d(np.asarray(c_prev, dtype=float)).ravel()
    if x.size == 0 or h.size == 0:
        raise ValueError("geron_lstm: x_t and h_prev must be non-empty")
    if c.size != h.size:
        raise ValueError(f"geron_lstm: c_prev has {c.size} units but h_prev has {h.size}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(h)) or not np.all(np.isfinite(c)):
        raise ValueError("geron_lstm: x_t, h_prev and c_prev must be finite")
    n_in, n_units = x.size, h.size

    W = {}
    for key in _KEYS:
        try:
            arr = np.asarray(weights[key], dtype=float)
        except (KeyError, TypeError):
            raise ValueError(f"geron_lstm: weights is missing required key {key!r}") from None
        if key.startswith("b"):
            arr = np.atleast_1d(arr).ravel()
            want = (n_units,)
        elif key.startswith("W"):
            arr = np.atleast_2d(arr)
            want = (n_units, n_in)
        else:
            arr = np.atleast_2d(arr)
            want = (n_units, n_units)
        if arr.shape != want:
            raise ValueError(f"geron_lstm: weights[{key!r}] has shape {arr.shape}, expected {want}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"geron_lstm: weights[{key!r}] contains non-finite values")
        W[key] = arr

    i = _sigmoid(W["W_i"] @ x + W["U_i"] @ h + W["b_i"])
    f = _sigmoid(W["W_f"] @ x + W["U_f"] @ h + W["b_f"])
    o = _sigmoid(W["W_o"] @ x + W["U_o"] @ h + W["b_o"])
    g = np.tanh(W["W_g"] @ x + W["U_g"] @ h + W["b_g"])
    c_t = f * c + i * g
    h_t = o * np.tanh(c_t)

    return RichResult(
        title="LSTM cell",
        summary_lines=[
            ("Units", n_units),
            ("Mean forget gate", float(np.mean(f))),
            ("Mean input gate", float(np.mean(i))),
            ("Mean output gate", float(np.mean(o))),
        ],
        interpretation=(
            "The cell state is updated only by scaling and adding, so gradients along it decay "
            "as a product of forget gates instead of a product of weight matrices."
        ),
        payload={
            "h_t": h_t,
            "c_t": c_t,
            "i_t": i,
            "f_t": f,
            "o_t": o,
            "g_t": g,
            "estimate": float(np.linalg.norm(h_t)),
            "n": int(n_units),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlstm: LSTM step c_t = f*c_{t-1} + i*g, h_t = o*tanh(c_t) with all four gates returned"


# compact alias per ledger/NAMING.md
geronlstm = geron_lstm
