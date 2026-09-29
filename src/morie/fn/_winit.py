# morie.fn -- function file (rootcoder007/morie)
"""Shared kernel of the weight-initialisation front ends (``vctrs``, ``xavir``, ``xvrig``).

Draws with the Philox generator (``_rng``) so the R twins in
``R/WeightInit.R`` produce the same matrices bit for bit (up to the
orthogonalisation rounding).
"""

from __future__ import annotations

import math

from ._qpcore import ssum
from ._rng import random_normal, random_uniform

__all__: list = []

SCALE = {
    "xavier_uniform": lambda fi, fo: math.sqrt(6.0 / (fi + fo)),
    "xavier_normal": lambda fi, fo: math.sqrt(2.0 / (fi + fo)),
    "he_uniform": lambda fi, fo: math.sqrt(6.0 / fi),
    "he_normal": lambda fi, fo: math.sqrt(2.0 / fi),
    "lecun_normal": lambda fi, fo: math.sqrt(1.0 / fi),
}

VARIANCE = {
    "xavier_uniform": lambda fi, fo: 2.0 / (fi + fo),
    "xavier_normal": lambda fi, fo: 2.0 / (fi + fo),
    "he_uniform": lambda fi, fo: 2.0 / fi,
    "he_normal": lambda fi, fo: 2.0 / fi,
    "lecun_normal": lambda fi, fo: 1.0 / fi,
    "orthogonal": lambda fi, fo: 1.0 / max(fi, fo),
}


def _aslist(v):
    return [float(t) for t in (v.tolist() if hasattr(v, "tolist") else v)]


def _mgs_columns(M):
    """Orthonormalise the columns of ``M`` (rows x cols, rows >= cols) by modified Gram-Schmidt."""
    r, c = len(M), len(M[0])
    Q = [[M[i][j] for i in range(r)] for j in range(c)]
    for j in range(c):
        for k in range(j):
            d = ssum(a * b for a, b in zip(Q[k], Q[j]))
            Q[j] = [a - d * b for a, b in zip(Q[j], Q[k])]
        nrm = math.sqrt(ssum(a * a for a in Q[j]))
        Q[j] = [a / nrm for a in Q[j]]
    return [[Q[j][i] for j in range(c)] for i in range(r)]


def draw(rows, cols, method, gain, seed, fan_in, fan_out):
    """``rows x cols`` weights, filled row by row from one Philox stream."""
    k = rows * cols
    if method in ("xavier_uniform", "he_uniform"):
        a = gain * SCALE[method](fan_in, fan_out)
        u = _aslist(random_uniform(k, seed=seed))
        v = [-a + 2.0 * a * t for t in u]
    elif method in ("xavier_normal", "he_normal", "lecun_normal"):
        s = gain * SCALE[method](fan_in, fan_out)
        v = [s * t for t in _aslist(random_normal(k, seed=seed))]
    elif method == "orthogonal":
        z = _aslist(random_normal(k, seed=seed))
        M = [z[i * cols : (i + 1) * cols] for i in range(rows)]
        if rows >= cols:
            Q = _mgs_columns(M)
        else:
            Qt = _mgs_columns([list(c) for c in zip(*M)])
            Q = [list(c) for c in zip(*Qt)]
        return [[gain * v for v in row] for row in Q]
    else:
        raise ValueError(f"Unknown method: {method}")
    return [v[i * cols : (i + 1) * cols] for i in range(rows)]


def moments(W):
    flat = [v for r in W for v in r]
    m = ssum(flat) / len(flat)
    return m, ssum((v - m) ** 2 for v in flat) / len(flat)
