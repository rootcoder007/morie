# morie.fn -- function file (rootcoder007/morie)
"""Layer normalisation."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['vitlnorm', 'vit_layer_norm']


def vitlnorm(x, gamma=None, beta=None):
    """Layer normalisation.

    Normalisation is over the hidden units of one sample, not over the batch, so every sample carries its own statistics and the layer behaves identically at batch size one and at batch size a thousand. Two details are taken from the paper rather than from common practice: the variance divides by H, not H-1, and there is no epsilon in the denominator. Adding one would change every output in the last digits and quietly break parity against the published definition, so a zero-variance row raises instead.


    Formula: mu = (1/H) sum_i a_i, sigma = sqrt((1/H) sum_i (a_i - mu)^2), y = gamma (a - mu)/sigma + beta

    Parameters
    ----------
    x : array-like, shape (n, H)
        One sample per row; a flat sequence is treated as one sample.
    gamma : array-like, optional
        Per-unit gain; ones if omitted.
    beta : array-like, optional
        Per-unit bias; zeros if omitted.

    Returns
    -------
    RichResult
        ``y``, ``mu``, ``sigma``, ``n``, ``H``.

    References
    ----------
    Ba, Kiros and Hinton (2016), Layer Normalization, arXiv:1607.06450,
    equation (3).  Verified against the paper.
    """
    X = C.mat(x) if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)) \
        else [C.vec(x)]
    n = len(X); H = len(X[0])
    g = C.vec(gamma) if gamma is not None else [1.0] * H
    b = C.vec(beta) if beta is not None else [0.0] * H
    if len(g) != H or len(b) != H:
        raise ValueError("gamma and beta must have length H")
    mus, sds, Y = [], [], []
    for row in X:
        mu = sum(row) / H
        s = math.sqrt(sum((v - mu) ** 2 for v in row) / H)
        if s <= 0:
            raise ValueError("a row has zero variance; layer norm is undefined")
        mus.append(mu); sds.append(s)
        Y.append([g[j] * (row[j] - mu) / s + b[j] for j in range(H)])
    return RichResult(payload={
        "y": Y, "mu": mus, "sigma": sds, "n": n, "H": H,
        "method": "Layer normalisation"})


vit_layer_norm = vitlnorm


def cheatsheet():
    return "vitlrn: Layer normalisation."
