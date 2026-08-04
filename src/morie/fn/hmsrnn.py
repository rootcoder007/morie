# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Simple RNN forward pass over a sequence."""

from . import _array_core as np

from ._richresult import RichResult
from .hmtanh import geron_tanh

__all__ = ["geron_simple_rnn"]


def geron_simple_rnn(X, Wx, Wh, b=None, h0=None):
    """
    Simple RNN forward pass over a sequence.

    Formula: h_t = tanh(W_x x_t + W_h h_{t-1} + b)

    Unrolled in time, one step at a time; the activation is delegated to
    :func:`morie.fn.hmtanh.geron_tanh`, which also supplies the per-step
    derivative ``1 - h_t^2``. The product of those derivatives along the
    sequence is returned as `jacobian_gain`: it is the factor that makes
    backpropagation through time vanish (gain << 1) or explode (gain >> 1).

    Parameters
    ----------
    X : array-like
        Sequence of inputs, shape (T, n_inputs). A 1-D input is read as
        T steps of one feature.
    Wx : array-like
        Input weights, shape (n_inputs, n_units).
    Wh : array-like
        Recurrent weights, shape (n_units, n_units).
    b : array-like, optional
        Bias of length n_units; default zeros.
    h0 : array-like, optional
        Initial hidden state; default zeros.

    Returns
    -------
    result : RichResult
        Keys: H, h_T, grads, jacobian_gain, estimate, n, method.

    Examples
    --------
    With no recurrence (W_h = 0) each step is just tanh of its own input:

    >>> r = geron_simple_rnn([[0.0], [1.0]], [[1.0]], [[0.0]])
    >>> [round(float(v), 6) for v in r["H"].ravel()]
    [0.0, 0.761594]

    Recurrence accumulates: with W_x = W_h = 1 and inputs 1, 0 the second
    state is tanh(tanh(1)) = tanh(0.761594):

    >>> r2 = geron_simple_rnn([[1.0], [0.0]], [[1.0]], [[1.0]])
    >>> [round(float(v), 6) for v in r2["H"].ravel()]
    [0.761594, 0.642015]
    >>> [round(float(v), 6) for v in r2["h_T"]]
    [0.642015]

    References
    ----------
    Géron Ch 13
    """
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    if Xa.ndim != 2 or Xa.size == 0:
        raise ValueError("geron_simple_rnn: X must be a non-empty (T, n_inputs) sequence")
    Wxa = np.asarray(Wx, dtype=float)
    if Wxa.ndim == 1:
        Wxa = Wxa.reshape(-1, 1)
    Wha = np.asarray(Wh, dtype=float)
    if Wha.ndim == 1:
        Wha = Wha.reshape(1, -1)
    if Wxa.ndim != 2 or Wha.ndim != 2:
        raise ValueError("geron_simple_rnn: Wx and Wh must be 2-D weight matrices")
    if Wxa.shape[0] != Xa.shape[1]:
        raise ValueError(f"geron_simple_rnn: X has {Xa.shape[1]} inputs but Wx has {Wxa.shape[0]} rows")
    n_units = Wxa.shape[1]
    if Wha.shape != (n_units, n_units):
        raise ValueError(f"geron_simple_rnn: Wh must be square ({n_units}, {n_units}), got {Wha.shape}")
    bias = np.zeros(n_units) if b is None else np.asarray(b, dtype=float).ravel()
    if bias.size != n_units:
        raise ValueError(f"geron_simple_rnn: b has {bias.size} entries but there are {n_units} units")
    h = np.zeros(n_units) if h0 is None else np.asarray(h0, dtype=float).ravel()
    if h.size != n_units:
        raise ValueError(f"geron_simple_rnn: h0 has {h.size} entries but there are {n_units} units")
    for nm, A in (("X", Xa), ("Wx", Wxa), ("Wh", Wha), ("b", bias), ("h0", h)):
        if not np.all(np.isfinite(A)):
            raise ValueError(f"geron_simple_rnn: {nm} contains non-finite values")

    T = Xa.shape[0]
    H = np.empty((T, n_units))
    G = np.empty((T, n_units))
    for t in range(T):
        act = geron_tanh(Xa[t] @ Wxa + h @ Wha + bias)
        h = np.asarray(act["a"], dtype=float)
        H[t] = h
        G[t] = np.asarray(act["grad"], dtype=float)
    gain = float(np.prod(np.max(G, axis=1)) * np.linalg.norm(Wha, 2) ** (T - 1)) if T > 1 else float(np.max(G))

    return RichResult(
        title="Simple RNN forward pass",
        summary_lines=[("Time steps", T), ("Units", int(n_units)), ("BPTT gain bound", gain)],
        interpretation=(
            "tanh derivatives are at most 1, so the BPTT gain is bounded by ||W_h||^(T-1): "
            "a spectral norm below 1 makes gradients vanish, above 1 makes them explode."
        ),
        payload={
            "H": H,
            "h_T": H[-1],
            "grads": G,
            "jacobian_gain": gain,
            "n_units": int(n_units),
            "estimate": float(np.mean(H[-1])),
            "n": int(T),
            "method": "Simple (Elman) RNN unrolled with tanh states from hmtanh",
        },
    )


def cheatsheet():
    return "hmsrnn: Simple RNN forward pass over a sequence"


# compact alias per ledger/NAMING.md
geronsimplernn = geron_simple_rnn
