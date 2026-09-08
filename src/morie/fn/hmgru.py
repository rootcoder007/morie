# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GRU cell: reset and update gates."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_gru"]

_METHOD = "GRU cell forward step"

_KEYS = ("W_z", "U_z", "b_z", "W_r", "U_r", "b_r", "W_h", "U_h", "b_h")


def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def geron_gru(x_t, h_prev, weights):
    """
    GRU cell: reset and update gates.

    Formula: z_t = sigma(...); r_t = sigma(...); h_t = (1-z_t)*h_{t-1} + z_t*h_tilde

    A single GRU time step.  The gates are

    ``z_t = sigma(W_z x_t + U_z h_{t-1} + b_z)``
    ``r_t = sigma(W_r x_t + U_r h_{t-1} + b_r)``
    ``h~_t = tanh(W_h x_t + U_h (r_t * h_{t-1}) + b_h)``
    ``h_t = (1 - z_t) * h_{t-1} + z_t * h~_t``

    with the update-gate convention of the spec: ``z_t`` is the weight
    on the *candidate*, so ``z_t = 0`` copies the previous state through
    unchanged.  The reset gate multiplies ``h_{t-1}`` inside the
    candidate only, which is the whole difference from an LSTM's
    separate cell state.

    Parameters
    ----------
    x_t : array-like, shape (n_in,)
        Input at this step.
    h_prev : array-like, shape (n_units,)
        Previous hidden state.
    weights : mapping
        Must supply W_z, U_z, b_z, W_r, U_r, b_r, W_h, U_h, b_h with
        W_* of shape (n_units, n_in), U_* of shape (n_units, n_units)
        and b_* of shape (n_units,).

    Returns
    -------
    result : RichResult
        Keys: h_t, z_t, r_t, h_tilde, estimate, n, method.

    Examples
    --------
    All-zero weights make every gate ``sigma(0) = 0.5`` and the
    candidate ``tanh(0) = 0``, so ``h_t = (1 - 0.5) * h_prev``:

    >>> import numpy as np
    >>> W = {k: (np.zeros((2, 2)) if k[0] in "WU" else np.zeros(2)) for k in
    ...      ("W_z", "U_z", "b_z", "W_r", "U_r", "b_r", "W_h", "U_h", "b_h")}
    >>> r = geron_gru([0.0, 0.0], [4.0, -2.0], W)
    >>> [float(v) for v in r["z_t"]]
    [0.5, 0.5]
    >>> [float(v) for v in r["h_t"]]
    [2.0, -1.0]

    A large negative update bias shuts the candidate out entirely, so
    the state is carried through:

    >>> W2 = dict(W); W2["b_z"] = np.array([-40.0, -40.0])
    >>> r2 = geron_gru([0.0, 0.0], [4.0, -2.0], W2)
    >>> [round(float(v), 9) for v in r2["h_t"]]
    [4.0, -2.0]

    References
    ----------
    Géron Ch 13
    """
    if not hasattr(weights, "__getitem__"):
        raise ValueError("geron_gru: weights must be a mapping with keys " + ", ".join(_KEYS))
    x = np.atleast_1d(np.asarray(x_t, dtype=float)).ravel()
    h = np.atleast_1d(np.asarray(h_prev, dtype=float)).ravel()
    if x.size == 0 or h.size == 0:
        raise ValueError("geron_gru: x_t and h_prev must be non-empty")
    n_in, n_units = x.size, h.size

    W = {}
    for key in _KEYS:
        try:
            arr = np.asarray(weights[key], dtype=float)
        except (KeyError, TypeError):
            raise ValueError(f"geron_gru: weights is missing required key {key!r}") from None
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
            raise ValueError(f"geron_gru: weights[{key!r}] has shape {arr.shape}, expected {want}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"geron_gru: weights[{key!r}] contains non-finite values")
        W[key] = arr
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(h)):
        raise ValueError("geron_gru: x_t and h_prev must be finite")

    z = _sigmoid(W["W_z"] @ x + W["U_z"] @ h + W["b_z"])
    r = _sigmoid(W["W_r"] @ x + W["U_r"] @ h + W["b_r"])
    h_tilde = np.tanh(W["W_h"] @ x + W["U_h"] @ (r * h) + W["b_h"])
    h_t = (1.0 - z) * h + z * h_tilde

    return RichResult(
        title="GRU cell",
        summary_lines=[
            ("Units", n_units),
            ("Mean update gate", float(np.mean(z))),
            ("Mean reset gate", float(np.mean(r))),
        ],
        interpretation=(
            "z near 0 carries the previous state forward unchanged; z near 1 replaces it with the candidate."
        ),
        payload={
            "h_t": h_t,
            "z_t": z,
            "r_t": r,
            "h_tilde": h_tilde,
            "estimate": float(np.linalg.norm(h_t)),
            "n": int(n_units),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmgru: GRU step h_t = (1-z)*h_{t-1} + z*tanh(W_h x + U_h (r*h) + b_h)"


# compact alias per ledger/NAMING.md
gerongru = geron_gru
