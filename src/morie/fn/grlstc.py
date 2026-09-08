# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LSTM cell forward pass."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_lstm_cell"]

_METHOD = "LSTM cell forward pass"


def _sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def geron_lstm_cell(x_t, h_prev, c_prev, Wf, Wi, Wg, Wo, bf, bi, bg, bo):
    r"""One LSTM step.

    .. math::
        f &= \sigma(W_f [h, x] + b_f) \\
        i &= \sigma(W_i [h, x] + b_i) \\
        g &= \tanh (W_g [h, x] + b_g) \\
        o &= \sigma(W_o [h, x] + b_o) \\
        c &= f \odot c_{\text{prev}} + i \odot g,\qquad
        h = o \odot \tanh c

    The cell state ``c`` is updated by *addition*, gated by ``f``.  That
    is the entire point of the architecture: the path from ``c_prev`` to
    ``c`` has derivative ``f`` rather than a matrix product, so as long
    as the forget gate stays open the gradient neither explodes nor
    vanishes over long spans -- which the plain RNN cannot manage.

    Each ``W`` acts on the concatenation ``[h_prev, x_t]``, so it has
    shape ``(H, H + n)``.

    Parameters
    ----------
    x_t : array-like, shape (n,)
    h_prev, c_prev : array-like, shape (H,)
    Wf, Wi, Wg, Wo : array-like, shape (H, H + n)
    bf, bi, bg, bo : array-like, shape (H,) or scalar

    Returns
    -------
    RichResult
        Payload keys ``h``, ``c``, ``f``, ``i``, ``g``, ``o``,
        ``forget_open`` (mean of ``f``), ``estimate`` (= ``h``), ``n``,
        ``method``.

    References
    ----------
    Géron Ch 13, LSTM section (Hochreiter and Schmidhuber 1997).

    Examples
    --------
    With zero weights and biases every gate sits at 0.5 and the
    candidate is 0, so the cell simply halves whatever it was
    remembering:

    >>> Z = [[0.0, 0.0]]
    >>> r = geron_lstm_cell([0.0], [0.0], [1.0], Z, Z, Z, Z, 0.0, 0.0, 0.0, 0.0)
    >>> r["f"], r["i"], r["g"]
    ([0.5], [0.5], [0.0])
    >>> r["c"]
    [0.5]
    >>> round(r["h"][0], 10)
    0.2310585786

    Slam the forget gate shut (a large negative bias) and the memory is
    wiped in one step:

    >>> r2 = geron_lstm_cell([0.0], [0.0], [1.0], Z, Z, Z, Z,
    ...                      -50.0, 0.0, 0.0, 0.0)
    >>> round(r2["c"][0], 12)
    0.0
    """
    x = np.asarray(x_t, dtype=float).ravel()
    h = np.asarray(h_prev, dtype=float).ravel()
    c = np.asarray(c_prev, dtype=float).ravel()
    if h.size != c.size:
        raise ValueError(f"h_prev has {h.size} units but c_prev has {c.size}.")
    H, n = h.size, x.size
    if H == 0:
        raise ValueError("h_prev is empty; the cell needs at least one unit.")
    z = np.concatenate([h, x])

    gates = {}
    for name, W, b in (("f", Wf, bf), ("i", Wi, bi), ("g", Wg, bg), ("o", Wo, bo)):
        M = np.atleast_2d(np.asarray(W, dtype=float))
        if M.shape != (H, H + n):
            raise ValueError(
                f"W{name} must have shape (H, H + n) = ({H}, {H + n}) to act on "
                f"[h_prev, x_t], got {M.shape}."
            )
        bv = np.asarray(b, dtype=float).ravel()
        if bv.size == 1:
            bv = np.full(H, float(bv[0]))
        if bv.size != H:
            raise ValueError(f"b{name} must be a scalar or length {H}, got {bv.size}.")
        if not np.all(np.isfinite(M)) or not np.all(np.isfinite(bv)):
            raise ValueError(f"W{name} and b{name} must be finite.")
        gates[name] = M @ z + bv
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(h)) and np.all(np.isfinite(c))):
        raise ValueError("x_t, h_prev and c_prev must be finite.")

    f = _sigmoid(gates["f"])
    i = _sigmoid(gates["i"])
    g = np.tanh(gates["g"])
    o = _sigmoid(gates["o"])
    c_new = f * c + i * g
    h_new = o * np.tanh(c_new)

    return RichResult(
        title="LSTM cell",
        summary_lines=[("Units", int(H)), ("Forget gate mean", float(f.mean()))],
        payload={
            "h": h_new.tolist(),
            "c": c_new.tolist(),
            "f": f.tolist(),
            "i": i.tolist(),
            "g": g.tolist(),
            "o": o.tolist(),
            "forget_open": float(f.mean()),
            "estimate": h_new.tolist(),
            "n": int(H),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grlstc: c = f*c_prev + i*g, h = o*tanh(c); additive cell path is why gradients survive"


# compact alias per ledger/NAMING.md
geronlstmcell = geron_lstm_cell
