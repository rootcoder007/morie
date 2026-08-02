# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Peephole LSTM cell: the gates see the cell state as well."""

from . import _array_core as np

from ._richresult import RichResult
from .grsig import geron_sigmoid

__all__ = ["geron_peephole_lstm_cell"]

_METHOD = "Peephole LSTM cell"


def _sig(z):
    return np.asarray(geron_sigmoid(z)["sigma"], dtype=float).reshape(np.shape(z))


def geron_peephole_lstm_cell(x_t, h_prev, c_prev, Wf, Wi, Wg, Wo, Uf, Ui, Uo, bf, bi, bg, bo):
    r"""LSTM step with peephole connections on the three gates.

    .. math::
        f_t &= \sigma(W_f[h_{t-1}, x_t] + U_f \odot c_{t-1} + b_f)\\
        i_t &= \sigma(W_i[h_{t-1}, x_t] + U_i \odot c_{t-1} + b_i)\\
        g_t &= \tanh(W_g[h_{t-1}, x_t] + b_g)\\
        c_t &= f_t \odot c_{t-1} + i_t \odot g_t\\
        o_t &= \sigma(W_o[h_{t-1}, x_t] + U_o \odot c_t + b_o)\\
        h_t &= o_t \odot \tanh(c_t)

    Two details that are easy to get wrong and are enforced here.  The
    output gate peeps at the *new* cell state :math:`c_t`, while the
    forget and input gates peep at the old :math:`c_{t-1}` -- Gers &
    Schmidhuber's ordering, and the reason the output gate can react to
    what was just written.  And the candidate :math:`g_t` has no peephole
    at all: it is content, not a gate.  Peepholes are diagonal, so ``Uf``,
    ``Ui`` and ``Uo`` are vectors of length ``n_h``, not matrices.

    Parameters
    ----------
    x_t : array-like, shape (n_in,)
    h_prev, c_prev : array-like, shape (n_h,)
    Wf, Wi, Wg, Wo : array-like, shape (n_h, n_h + n_in)
        Applied to the concatenation ``[h_prev, x_t]``.
    Uf, Ui, Uo : array-like, shape (n_h,)
        Diagonal peephole weights.
    bf, bi, bg, bo : array-like, shape (n_h,)

    Returns
    -------
    RichResult
        Payload keys ``h``, ``c``, ``f``, ``i``, ``g``, ``o``,
        ``estimate`` (h), ``n``, ``method``.

    References
    ----------
    Géron Ch 13, Peephole LSTM section.

    Examples
    --------
    All weights zero except a peephole on the forget gate, ``c_prev = 1``:
    the forget gate sees ``sigma(2) = 0.880797`` where a plain LSTM would
    see ``sigma(0) = 0.5``.

    >>> W = [[0.0, 0.0]]
    >>> r = geron_peephole_lstm_cell([0.0], [0.0], [1.0], W, W, W, W,
    ...                              [2.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0])
    >>> round(r["f"][0], 6)
    0.880797
    >>> round(r["c"][0], 6)
    0.880797

    The output gate peeps at the new cell state, so it is not 0.5 either:

    >>> r2 = geron_peephole_lstm_cell([0.0], [0.0], [1.0], W, W, W, W,
    ...                               [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0])
    >>> round(r2["c"][0], 6)
    0.5
    >>> round(r2["o"][0], 6)
    0.622459
    """
    x = np.asarray(x_t, dtype=float).ravel()
    h = np.asarray(h_prev, dtype=float).ravel()
    c = np.asarray(c_prev, dtype=float).ravel()
    n_h = h.size
    if n_h == 0 or x.size == 0:
        raise ValueError("x_t and h_prev must both be non-empty.")
    if c.size != n_h:
        raise ValueError(f"c_prev has {c.size} units but h_prev has {n_h}.")
    n_cat = n_h + x.size
    W = {}
    for name, M in (("Wf", Wf), ("Wi", Wi), ("Wg", Wg), ("Wo", Wo)):
        A = np.atleast_2d(np.asarray(M, dtype=float))
        if A.shape != (n_h, n_cat):
            raise ValueError(
                f"{name} must be ({n_h}, {n_cat}) to act on [h_prev, x_t], got {A.shape}."
            )
        W[name] = A
    U = {}
    for name, v in (("Uf", Uf), ("Ui", Ui), ("Uo", Uo)):
        a = np.asarray(v, dtype=float).ravel()
        if a.size != n_h:
            raise ValueError(
                f"{name} is a diagonal peephole and must have {n_h} entries, got {a.size}."
            )
        U[name] = a
    B = {}
    for name, v in (("bf", bf), ("bi", bi), ("bg", bg), ("bo", bo)):
        a = np.asarray(v, dtype=float).ravel()
        if a.size != n_h:
            raise ValueError(f"{name} must have {n_h} entries, got {a.size}.")
        B[name] = a
    for name, M in list(W.items()) + list(U.items()) + list(B.items()) + [("x_t", x), ("h_prev", h), ("c_prev", c)]:
        if not np.all(np.isfinite(M)):
            raise ValueError(f"{name} contains non-finite values.")

    z = np.concatenate([h, x])
    f = _sig(W["Wf"] @ z + U["Uf"] * c + B["bf"])
    i = _sig(W["Wi"] @ z + U["Ui"] * c + B["bi"])
    g = np.tanh(W["Wg"] @ z + B["bg"])
    c_new = f * c + i * g
    o = _sig(W["Wo"] @ z + U["Uo"] * c_new + B["bo"])
    h_new = o * np.tanh(c_new)

    return RichResult(
        title="Peephole LSTM cell",
        summary_lines=[("Units", n_h), ("Mean forget gate", float(f.mean()))],
        payload={
            "h": h_new.tolist(),
            "c": c_new.tolist(),
            "f": f.tolist(),
            "i": i.tolist(),
            "g": g.tolist(),
            "o": o.tolist(),
            "estimate": h_new.tolist(),
            "n": int(n_h),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grpels: f,i peep at c_{t-1}, o peeps at c_t, g has no peephole; U* are diagonal vectors"
