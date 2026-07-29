# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.27: the LSTM's softmax over the vocabulary."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch6_lstm_softmax_word"]


def _softmax_logits(z):
    """Overflow-safe softmax; km104 imports this."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    if z.size == 0:
        raise ValueError("the vocabulary is empty.")
    if not np.all(np.isfinite(z)):
        raise ValueError("a logit is not finite.")
    e = np.exp(z - z.max())
    return e / e.sum()


def _hidden(f, c, name):
    if f is None:
        v = c
    elif callable(f):
        v = f(c)
    else:
        v = f
    v = np.atleast_1d(np.asarray(v, dtype=float))
    if v.size == 0:
        raise ValueError(f"{name} produced an empty vector.")
    return v


def kamath_ch6_lstm_softmax_word(U, f, c_t_1, b):
    """P(w_t = i | c_{t-1}) = exp(U_i^T f(c_{t-1}) + b_i) /
    sum_j exp(U_j^T f(c_{t-1}) + b_j).

    ``U`` is the (V, d) word representation matrix, ``f`` the LSTM
    output -- a callable applied to ``c_t_1``, the hidden vector
    itself, or None when ``c_t_1`` already IS that vector -- and ``b`` the (V,) bias. Adding a constant to every bias
    leaves the distribution unchanged, which the tests assert.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.27, printed
    p. 253.

    Examples
    --------
    >>> out = kamath_ch6_lstm_softmax_word(
    ...     [[1.0, 0.0], [0.0, 1.0]], None, [1.0, 0.0], [0.0, 0.0])
    >>> round(out["p"][0], 10)
    0.7310585786
    >>> out["argmax"]
    0
    """
    Um = np.atleast_2d(np.asarray(U, dtype=float))
    h = _hidden(f, c_t_1, "f")
    bv = np.atleast_1d(np.asarray(b, dtype=float))
    if Um.shape[1] != h.shape[0]:
        raise ValueError(
            f"U has width {Um.shape[1]} but the hidden vector has "
            f"{h.shape[0]}.")
    if bv.shape[0] != Um.shape[0]:
        raise ValueError(
            f"b has {bv.shape[0]} entries but the vocabulary has "
            f"{Um.shape[0]}.")
    logits = Um @ h + bv
    p = _softmax_logits(logits)
    return RichResult(payload={
        "p": [float(v) for v in p], "logits": [float(v) for v in logits],
        "argmax": int(np.argmax(p)), "estimate": float(p.max()),
        "n": int(p.size),
        "method": "LSTM vocabulary softmax (Kamath Eq 6.27)"})


def cheatsheet():
    return "km103: softmax(U f(c_{t-1}) + b) over the vocabulary"
