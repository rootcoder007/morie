# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 6.28: Affect-LM's affect-shifted vocabulary softmax."""

from . import _array_core as np

from ._richresult import RichResult
from .km103 import _hidden, _softmax_logits, kamath_ch6_lstm_softmax_word

__all__ = ["kamath_ch6_affect_lm"]


def kamath_ch6_affect_lm(U, V, f, g, c, e, beta, b):
    """P(w_t = i | c_{t-1}, e_{t-1}) = softmax(U_i^T f(c) +
    beta V_i^T g(e) + b_i).

    Eq 6.27 with one extra term: the affect category's contribution,
    scaled by the affect strength beta. beta = 0 collapses it EXACTLY
    to km103's distribution -- the tests check that composition
    identity -- and large beta drives generation toward the affect
    category, which is how the vocabulary shift detoxifies.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 6, Eq 6.28, printed
    p. 253.

    Examples
    --------
    >>> U = [[1.0, 0.0], [0.0, 1.0]]
    >>> Vm = [[0.0, 0.0], [1.0, 0.0]]
    >>> out = kamath_ch6_affect_lm(U, Vm, None, None, [1.0, 0.0],
    ...                            [1.0, 0.0], 2.0, [0.0, 0.0])
    >>> round(out["p"][1], 10)
    0.7310585786
    """
    Um = np.atleast_2d(np.asarray(U, dtype=float))
    Vm = np.atleast_2d(np.asarray(V, dtype=float))
    hc = _hidden(f, c, "f")
    he = _hidden(g, e, "g")
    bv = np.atleast_1d(np.asarray(b, dtype=float))
    beta = float(beta)
    if not np.isfinite(beta):
        raise ValueError("beta must be finite.")
    if Um.shape[1] != hc.shape[0]:
        raise ValueError(
            f"U has width {Um.shape[1]} but f(c) has {hc.shape[0]}.")
    if Vm.shape[1] != he.shape[0]:
        raise ValueError(
            f"V has width {Vm.shape[1]} but g(e) has {he.shape[0]}.")
    if Vm.shape[0] != Um.shape[0]:
        raise ValueError(
            f"V covers {Vm.shape[0]} words but U covers {Um.shape[0]}.")
    if bv.shape[0] != Um.shape[0]:
        raise ValueError(
            f"b has {bv.shape[0]} entries but the vocabulary has "
            f"{Um.shape[0]}.")
    base = Um @ hc
    affect = beta * (Vm @ he)
    p = _softmax_logits(base + affect + bv)
    return RichResult(payload={
        "p": [float(v) for v in p],
        "affect_term": [float(v) for v in affect],
        "argmax": int(np.argmax(p)), "beta": beta,
        "estimate": float(p.max()), "n": int(p.size),
        "method": "Affect-LM vocabulary softmax (Kamath Eq 6.28)"})


def cheatsheet():
    return "km104: km103's softmax plus beta V g(e), affect shift"
