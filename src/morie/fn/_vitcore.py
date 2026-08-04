# morie.fn -- slice s01 (rootcoder007/morie)
"""Shared primitives for the Vision Transformer modules.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S.,
Uszkoreit, J. and Houlsby, N. (2021), "An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale", ICLR 2021; arXiv:2010.11929v2.

Everything here is infrastructure, not method: a deterministic stand-in
for the learned parameters, layer normalisation, and the channel
convention used to read an image.  The method itself lives in vitptm,
vitcls, vitatt, vitmlp, vitfwd and vitfsv.

The paper describes trained networks.  A reproducible reference
implementation cannot ship trained weights, so every parameter matrix
here is filled from the shared deterministic normal stream
(_s03core.normdraws on the base-2 van der Corput sequence, mirrored in R
by .s03normdraws).  Both language arms therefore hold the SAME numbers,
not merely numbers from the same distribution, which is what makes a
1e-9 parity comparison meaningful.  A single stream is consumed in a
documented order via the ``skip`` offset, so distinct weight matrices
are distinct.

Layer normalisation is Ba, Kiros and Hinton (2016), "Layer
Normalization", arXiv:1607.06450, which the paper cites as "LN" in
Eqs. (2)-(4), p. 4.  The paper does not state an epsilon; 1e-6 is used
here and is stated as this implementation's own choice, not as a
quotation.  Gain and bias are the identity (gamma = 1, beta = 0).
"""

from __future__ import annotations

import math

from . import _s03core as core

__all__: list[str] = []

LN_EPS = 1e-6


def draw(nr, nc, skip=0, scale=1.0):
    """A deterministic nr-by-nc parameter matrix, row-major off the stream."""
    nr = int(nr)
    nc = int(nc)
    skip = int(skip)
    if nr < 1 or nc < 1:
        raise ValueError("draw: matrix dimensions must be positive")
    if skip < 0:
        raise ValueError("draw: skip must be non-negative")
    d = core.normdraws(skip + nr * nc, 2)
    return [[float(scale) * d[skip + i * nc + j] for j in range(nc)] for i in range(nr)]


def layernorm(v, eps=LN_EPS):
    """LN(v) = (v - mean v) / sqrt(pop.var v + eps), gamma = 1, beta = 0."""
    n = len(v)
    if n == 0:
        raise ValueError("layernorm: empty vector")
    s = 0.0
    for x in v:
        s += x
    m = s / n
    q = 0.0
    for x in v:
        q += (x - m) * (x - m)
    sd = math.sqrt(q / n + eps)
    return [(x - m) / sd for x in v]


def layernorm_rows(A, eps=LN_EPS):
    return [layernorm(r, eps) for r in A]


def channels(image):
    """Read an image as a list of C matrices, each H-by-W.

    A plain H-by-W matrix is the single-channel case, C = 1.
    """
    if (
        isinstance(image, (list, tuple))
        and len(image) > 0
        and isinstance(image[0], (list, tuple))
        and len(image[0]) > 0
        and isinstance(image[0][0], (list, tuple))
    ):
        out = [core.mat(m) for m in image]
    else:
        out = [core.mat(image)]
    h = len(out[0])
    w = len(out[0][0])
    for m in out:
        if len(m) != h or any(len(r) != w for r in m):
            raise ValueError("channels: all channels must have the same H and W")
    return out


def argmax_first(v):
    """Index of the first maximum, 0-based (R's which.max, 1-based, matches)."""
    if not v:
        raise ValueError("argmax_first: empty vector")
    b = 0
    for i in range(1, len(v)):
        if v[i] > v[b]:
            b = i
    return b
