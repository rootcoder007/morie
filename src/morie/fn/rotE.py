# morie.fn -- slice s03 (rootcoder007/morie)
"""RotatE knowledge-graph embedding score.

Source consulted (FETCHED): Sun, Z., Deng, Z.-H., Nie, J.-Y. and Tang,
J. (2019).  RotatE: knowledge graph embedding by relational rotation in
complex space.  *ICLR* (arXiv:1902.10197).  The paper states, verbatim:
"Given a triplet (h, r, t), we expect that t = h (o) r, where h, r, t are
in C^k are the embeddings, the modulus |r_i| = 1 and (o) denotes the
Hadamard (element-wise) product", so the score function of its table 1
is

    d_r(h, t) = || h (o) r - t ||

with each relation entry a pure rotation, r_i = exp(i theta_i).  The
unit modulus is what makes the relation a rotation rather than a general
scaling, and it is enforced here rather than assumed: relations are
supplied as *phases* theta, not as free complex numbers.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["rotate"]


def rotate(triples, dim=None, h_re=None, h_im=None, theta=None, t_re=None,
           t_im=None, gamma=None):
    """RotatE distance for one triple, and the margin score if gamma is given.

    Parameters
    ----------
    triples : any
        Carried through; only used when the component vectors are absent,
        in which case it must be a flat list
        [h_re..., h_im..., theta..., t_re..., t_im...] of length 5 * dim.
    dim : int, optional
        The embedding dimension k.
    h_re, h_im : array-like, optional
        Real and imaginary parts of the head embedding.
    theta : array-like, optional
        Relation phases; the relation is exp(i theta), modulus 1.
    t_re, t_im : array-like, optional
        Real and imaginary parts of the tail embedding.
    gamma : float, optional
        Margin; the score gamma - d is then returned.

    Returns
    -------
    RichResult with payload:
        estimate : d_r(h, t)
        distance : same as estimate
        score    : gamma - d, nan when gamma is None
        per_dim  : the per-dimension modulus of h(o)r - t
    """
    if h_re is None:
        flat = k.vec(triples)
        d = int(dim) if dim is not None else len(flat) // 5
        h_re = flat[0:d]
        h_im = flat[d:2 * d]
        theta = flat[2 * d:3 * d]
        t_re = flat[3 * d:4 * d]
        t_im = flat[4 * d:5 * d]
    hr = k.vec(h_re)
    hi = k.vec(h_im)
    th = k.vec(theta)
    tr = k.vec(t_re)
    ti = k.vec(t_im)
    d = len(hr)
    per = []
    s = 0.0
    for j in range(d):
        cr = math.cos(th[j])
        si = math.sin(th[j])
        re = hr[j] * cr - hi[j] * si - tr[j]
        im = hr[j] * si + hi[j] * cr - ti[j]
        m = math.sqrt(re * re + im * im)
        per.append(m)
        s += m * m
    dist = math.sqrt(s)
    return RichResult(
        title="RotatE score",
        summary_lines=[("distance", dist)],
        payload={
            "estimate": dist,
            "distance": dist,
            "score": float(gamma) - dist if gamma is not None else float("nan"),
            "per_dim": per,
            "dim": d,
            "method": "RotatE d_r(h, t) = ||h o r - t|| with |r_i| = 1 (Sun et al. 2019)",
        },
    )


def cheatsheet():
    return "rotE: RotatE -- rotation in complex space"
