# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bidirectional RNN: concatenate forward and backward hidden states."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_bidirectional_rnn"]


def _phi(z, kind):
    if kind == "tanh":
        return np.tanh(z)
    if kind == "relu":
        return np.maximum(z, 0.0)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    if kind == "identity":
        return z
    raise ValueError(
        f"geron_bidirectional_rnn: unknown activation {kind!r}; expected tanh, relu, sigmoid or identity"
    )


def geron_bidirectional_rnn(X, Wx_f, Wh_f, Wx_b, Wh_b, b_f=None, b_b=None, h0_f=None, h0_b=None, activation="tanh"):
    """
    Bidirectional RNN: concatenate forward and backward hidden states.

    Formula: h_t = [h_t^fwd; h_t^bwd]

    The backward pass runs the sequence in reverse but its states are
    re-aligned to the original time index before concatenation, so
    ``output[t]`` mixes the prefix up to t with the suffix from t.

    Parameters
    ----------
    X : array-like, shape (T, d)
        Input sequence.
    Wx_f, Wx_b : array-like, shape (d, h)
        Input-to-hidden weights for each direction.
    Wh_f, Wh_b : array-like, shape (h, h)
        Recurrent weights for each direction.
    b_f, b_b : array-like, shape (h,), optional
        Biases; default zeros.
    h0_f, h0_b : array-like, shape (h,), optional
        Initial states; default zeros.
    activation : {"tanh", "relu", "sigmoid", "identity"}
        Elementwise nonlinearity.

    Returns
    -------
    result : RichResult
        Keys: output, h_fwd, h_bwd, final, estimate, n, method.

    Examples
    --------
    With no recurrence (Wh = 0) each direction is a per-step map, and the
    backward states line up reversed against the forward ones:

    >>> r = geron_bidirectional_rnn([[1.0], [0.0]], [[1.0]], [[0.0]], [[1.0]], [[0.0]])
    >>> [[round(float(v), 6) for v in row] for row in r["output"]]
    [[0.761594, 0.761594], [0.0, 0.0]]

    Turning on the forward recurrence accumulates history:

    >>> r2 = geron_bidirectional_rnn([[1.0], [1.0]], [[1.0]], [[1.0]], [[0.0]], [[0.0]])
    >>> [round(float(v), 6) for v in r2["h_fwd"].ravel()]
    [0.761594, 0.942681]

    References
    ----------
    Géron Ch 14
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"geron_bidirectional_rnn: X must be 2-D (T, d), got ndim={A.ndim}")
    T, d = A.shape
    if T == 0:
        raise ValueError("geron_bidirectional_rnn: X has no time steps")

    Wxf = np.asarray(Wx_f, dtype=float)
    Whf = np.asarray(Wh_f, dtype=float)
    Wxb = np.asarray(Wx_b, dtype=float)
    Whb = np.asarray(Wh_b, dtype=float)
    for name, M in (("Wx_f", Wxf), ("Wh_f", Whf), ("Wx_b", Wxb), ("Wh_b", Whb)):
        if M.ndim != 2:
            raise ValueError(f"geron_bidirectional_rnn: {name} must be a 2-D matrix, got ndim={M.ndim}")
    h = Wxf.shape[1]
    if Wxf.shape[0] != d or Wxb.shape[0] != d:
        raise ValueError(
            f"geron_bidirectional_rnn: X has {d} features but Wx_f/Wx_b expect "
            f"{Wxf.shape[0]}/{Wxb.shape[0]}"
        )
    if Wxb.shape[1] != h:
        raise ValueError(
            f"geron_bidirectional_rnn: forward hidden size {h} does not match backward {Wxb.shape[1]}"
        )
    if Whf.shape != (h, h) or Whb.shape != (h, h):
        raise ValueError(f"geron_bidirectional_rnn: Wh_f and Wh_b must both be ({h}, {h})")

    bf = np.zeros(h) if b_f is None else np.asarray(b_f, dtype=float).ravel()
    bb = np.zeros(h) if b_b is None else np.asarray(b_b, dtype=float).ravel()
    hf = np.zeros(h) if h0_f is None else np.asarray(h0_f, dtype=float).ravel()
    hb = np.zeros(h) if h0_b is None else np.asarray(h0_b, dtype=float).ravel()
    for name, vec in (("b_f", bf), ("b_b", bb), ("h0_f", hf), ("h0_b", hb)):
        if vec.size != h:
            raise ValueError(f"geron_bidirectional_rnn: {name} has {vec.size} entries but the hidden size is {h}")

    H_f = np.empty((T, h))
    for t in range(T):
        hf = _phi(A[t] @ Wxf + hf @ Whf + bf, activation)
        H_f[t] = hf

    H_b = np.empty((T, h))
    for t in range(T - 1, -1, -1):
        hb = _phi(A[t] @ Wxb + hb @ Whb + bb, activation)
        H_b[t] = hb

    out = np.hstack([H_f, H_b])

    return RichResult(
        title="Bidirectional RNN",
        summary_lines=[("Time steps", T), ("Hidden size per direction", h), ("Output width", 2 * h)],
        payload={
            "output": out,
            "h_fwd": H_f,
            "h_bwd": H_b,
            "final": np.concatenate([H_f[-1], H_b[0]]),
            "hidden_size": int(h),
            "estimate": float(np.mean(out)),
            "n": int(T),
            "method": "Bidirectional RNN with concatenated forward/backward states",
        },
    )


def cheatsheet():
    return "hmbrnn: Bidirectional RNN: concatenate forward and backward hidden states"
