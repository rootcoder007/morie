# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classification MLP output head: softmax over K classes."""

import numpy as np

from ._richresult import RichResult
from .grlinf import geron_linear_layer_forward

__all__ = ["geron_classification_mlp_output"]

_METHOD = "Classification MLP softmax head"


def _softmax_rows(Z):
    """Row-wise softmax, shifted by the row max so exp never overflows."""
    M = Z.max(axis=1, keepdims=True)
    E = np.exp(Z - M)
    return E / E.sum(axis=1, keepdims=True)


def geron_classification_mlp_output(a_last, W_out, b_out):
    r"""Softmax head for K-class classification.

    .. math::
        z = W_{\text{out}} a_{L-1} + b_{\text{out}},\qquad
        p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}

    The exponentials are taken after subtracting the row maximum.  That
    shift cancels exactly in the ratio, and without it logits of a few
    hundred -- ordinary late in training -- overflow to ``inf`` and the
    probabilities come back as ``nan``.

    The affine stage delegates to
    :func:`morie.fn.grlinf.geron_linear_layer_forward`.

    Parameters
    ----------
    a_last : array-like, shape (h,) or (m, h)
    W_out : array-like, shape (K, h)
    b_out : array-like, shape (K,) or scalar

    Returns
    -------
    RichResult
        Payload keys ``probabilities``, ``logits``, ``predicted_class``,
        ``max_probability``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 9, Classification MLPs section.

    Examples
    --------
    Logits ``[1, 0]`` give the familiar two-class split
    ``e/(e+1) = 0.731...``:

    >>> W = [[1.0], [0.0]]
    >>> r = geron_classification_mlp_output([1.0], W, [0.0, 0.0])
    >>> [round(p, 7) for p in r["probabilities"]]
    [0.7310586, 0.2689414]
    >>> r["predicted_class"]
    0

    Adding a constant to every logit changes nothing -- softmax is
    shift-invariant, and the implementation relies on that:

    >>> r2 = geron_classification_mlp_output([1.0], W, [500.0, 500.0])
    >>> [round(p, 7) for p in r2["probabilities"]]
    [0.7310586, 0.2689414]
    """
    inner = geron_linear_layer_forward(a_last, W_out, b_out)
    Z = np.atleast_2d(np.asarray(inner["output"], dtype=float))
    if Z.shape[1] < 2:
        raise ValueError(
            f"a softmax head needs at least 2 classes, got {Z.shape[1]}; "
            f"use grlogp for the single-logit binary case."
        )
    P = _softmax_rows(Z)
    single = np.asarray(inner["output"]).ndim == 1
    pred = P.argmax(axis=1)

    return RichResult(
        title="Classification MLP output (softmax)",
        summary_lines=[("Classes", int(Z.shape[1])), ("Instances", int(Z.shape[0]))],
        payload={
            "probabilities": P[0].tolist() if single else P.tolist(),
            "logits": Z[0].tolist() if single else Z.tolist(),
            "predicted_class": int(pred[0]) if single else pred.tolist(),
            "max_probability": float(P.max(axis=1)[0]) if single else P.max(axis=1).tolist(),
            "n_classes": int(Z.shape[1]),
            "estimate": P[0].tolist() if single else P.tolist(),
            "n": int(Z.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grmlc: p_k = softmax(W_out a + b_out), max-shifted (delegates to grlinf)"
