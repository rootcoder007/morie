# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kamath Eq 9.12: the masked object classification (MOC) loss."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_ch9_moc_loss"]


def kamath_ch9_moc_loss(theta, w, v, g_theta, labels=None):
    r"""L_MOC = E_(w,v) [ sum_{i=1..M} CE(c(v_m^i), g_theta(v_m^i)) ].

    ``g_theta`` is either the softmax distribution over the T object
    classes for each of the M masked regions (an M x T array) or a
    callable ``g_theta(v) -> that array``. ``labels`` (equivalently
    ``w`` when it carries them) are the ground-truth classes
    c(v_m^i), given as integer indices or one-hot rows.

    The book prints a leading minus in front of E[sum CE]; CE is
    itself the negative log-likelihood, so the loss MINIMIZED is the
    positive quantity returned in ``estimate``. The literal printed
    sign is kept available as ``as_printed``.

    References: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 9, Eq 9.12, printed
    p. 388.

    Examples
    --------
    >>> import math
    >>> out = kamath_ch9_moc_loss(None, None, [[0.0]],
    ...                           [[0.5, 0.5], [0.25, 0.75]],
    ...                           labels=[0, 1])
    >>> abs(out["estimate"] - (math.log(2) - math.log(0.75))) < 1e-12
    True
    """
    if theta is not None and not callable(theta):
        raise ValueError("theta must be a callable model or None.")
    G = g_theta(v) if callable(g_theta) else g_theta
    G = np.atleast_2d(np.asarray(G, dtype=float))
    if G.size == 0:
        raise ValueError("no class distributions were given for the "
                         "masked regions.")
    if np.any(G < 0) or np.any(G > 1):
        raise ValueError("g_theta must be a softmax distribution over "
                         "the object classes.")
    if np.max(np.abs(G.sum(axis=1) - 1.0)) > 1e-6:
        raise ValueError("each row of g_theta must sum to 1; it is a "
                         "distribution over the T object classes.")
    y = labels if labels is not None else w
    if y is None:
        raise ValueError("labels= (the ground-truth classes c(v_m^i)) "
                         "are required.")
    Y = np.asarray(y)
    if Y.ndim == 1:
        idx = Y.astype(int)
        if np.any((idx < 0) | (idx >= G.shape[1])):
            raise ValueError("a class index lies outside the "
                             f"{G.shape[1]} object classes.")
        if idx.size != G.shape[0]:
            raise ValueError(
                f"{idx.size} labels for {G.shape[0]} masked regions.")
        probs = G[np.arange(G.shape[0]), idx]
        onehot = np.zeros_like(G)
        onehot[np.arange(G.shape[0]), idx] = 1.0
    else:
        onehot = Y.astype(float)
        if onehot.shape != G.shape:
            raise ValueError(
                f"one-hot labels are {onehot.shape} but g_theta is "
                f"{G.shape}.")
        if not np.all((onehot == 0) | (onehot == 1)) or \
                np.any(onehot.sum(axis=1) != 1):
            raise ValueError("2-D labels must be one-hot rows, one "
                             "true class per masked region.")
        probs = (onehot * G).sum(axis=1)
    with np.errstate(divide="ignore"):
        ce = -np.log(probs)
    total = float(ce.sum())
    return RichResult(payload={
        "estimate": total, "as_printed": -total,
        "per_region": [float(c) for c in ce],
        "true_class_probability": [float(p) for p in probs],
        "n_masked_regions": int(G.shape[0]), "n": int(G.shape[0]),
        "method": "masked object classification loss (Kamath Eq 9.12)"})


def cheatsheet():
    return "km140: summed cross-entropy over masked image regions"
